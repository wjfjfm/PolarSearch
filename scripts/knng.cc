#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <queue>
#include <chrono>
#include <cassert>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <immintrin.h>
#include <math.h>
#include <ctime>

using namespace std;

#define FILE_D 200
#define D 208
#define K 100
#define HEAP_K 200
#define MAX_FLOAT 1e30
#define MAX_UINT32 0xffffffff
#define HASH_DEL 0x8fffffff
#define BATCH 256
#define THREAD 32
#define SLOTS 512
#define BITMAP 0x1ff
#define BITMAP_LOW 0xffff
#define SLOT_SHIFT 9
#define BUILD_BATCH 4096

#define LINEAR_HEAP 6000

#define DEBUG

uint64_t M = 0;
uint64_t BATCH_M;
float* vectors;
uint32_t* ids;
time_t end_time;
time_t start_time;
std::thread* load_thread;

atomic<bool> load_finished = false;
atomic<uint32_t> thread_running = 0;

class TopQue;
TopQue *topk_que;

struct dis_id_t {
  float dis;
  uint32_t id;
};

struct id2_dis_t {
  uint32_t id1;
  uint32_t id2;
  float dis;
};

struct heap_t {
  float heap_top;

  dis_id_t *heap;
  uint32_t *evicted;
  uint32_t evicted_size;
  uint32_t *slots;
  uint32_t size;
  uint32_t max_size;
  uint32_t insert_cnt;
  uint32_t clear_cnt;

  std::atomic_flag flag = ATOMIC_FLAG_INIT;
  bool seed;

  inline void lock() {
    while (flag.test_and_set(std::memory_order_acquire)) { }
  }

  inline void unlock() {
    flag.clear(std::memory_order_release);
  }

  void init(dis_id_t* _heap, uint32_t _max_size, uint32_t *_slots) {
    heap = _heap;
    max_size = _max_size;
    slots = _slots;
    evicted = _slots + SLOTS;
    evicted_size = 0;
    insert_cnt = 0;
    clear_cnt = 0;
    seed = false;
    flag.clear();

    heap[0].dis = MAX_FLOAT;
    heap[0].id = MAX_UINT32;
    heap_top = MAX_FLOAT;
    size = 1;
  }

  void reinit() {
    if (slots) memset(slots, 0xff, SLOTS * sizeof(uint32_t));
    heap[0].dis = MAX_FLOAT;
    heap[0].id = MAX_UINT32;
    heap_top = MAX_FLOAT;
    size = 1;
  }

  void heap_down(uint32_t index) {
    uint32_t left = 2 * index + 1;
    uint32_t right = 2 * index + 2;
    uint32_t largest = index;

    if (left < size && heap[left].dis > heap[largest].dis) {
      largest = left;
    }

    if (right < size && heap[right].dis > heap[largest].dis) {
      largest = right;
    }

    if (largest != index) {
      std::swap(heap[index], heap[largest]);
      heap_down(largest);
    }
  }

  void heap_up(uint32_t index) {
    uint32_t parent = (index - 1) / 2;
    while (index > 0 && heap[index].dis > heap[parent].dis) {
      std::swap(heap[index], heap[parent]);
      index = parent;
      parent = (index - 1) / 2;
    }
  }

  void insert_weak(float dis, uint32_t id) {
    if (size == max_size) {
      if (dis >= heap_top) { return; }

      heap[0].dis = dis;
      heap[0].id = id;
      heap_down(0);

      heap_top = heap[0].dis;
      return;
    }

    heap[size].dis = dis;
    heap[size].id = id;
    heap_up(size);
    size++;
  }

  inline bool hash_check_and_set(uint32_t id) {
    uint32_t &slot_value = slots[id & BITMAP];
    const uint32_t value = id >> SLOT_SHIFT;
    if (slot_value & BITMAP_LOW == value || (slot_value >> 16) == value) return false;
    slot_value = (slot_value << 16) + value;
    return true;
  }

  void insert_unique(float dis, uint32_t id) {
    lock();
    if (size == max_size) {
      if (dis >= heap_top) { unlock(); return; }
      if (!hash_check_and_set(id)) { unlock(); return; }

      if (evicted[evicted_size % K] != heap[0].id) {
        evicted_size++;
        evicted[evicted_size % K] = heap[0].id;
      }

      heap[0].dis = dis;
      heap[0].id = id;
      heap_down(0);
      heap_top = heap[0].dis;

      // insert_cnt++;

      unlock();
      return;
    }

    if (!hash_check_and_set(id)) { unlock(); return; }
    heap[size].dis = dis;
    heap[size].id = id;
    heap_up(size);
    size++;
    unlock();
  }

  uint32_t pop() {
    if (size == 0) return MAX_UINT32;

    uint32_t result = heap[0].id;
    while(heap[0].id == result) {
      size--;

      if (size > 0) {
        heap[0].id = heap[size].id;
        heap[0].dis = heap[size].dis;
        heap_down(0);
      } else {
        break;
      }
    }

    return result;
  }

  uint32_t pop_evicted() {
    if (evicted_size == 0) return MAX_UINT32;
    if (evicted[evicted_size % K] == heap_top) {
      evicted_size--;
      if (evicted_size == 0) return MAX_UINT32;
    }

    uint32_t result = evicted[evicted_size % K];
    evicted_size--;
    return result;
  }
};

class TopQue {
 public:
  TopQue(uint64_t M) {
    buf = (dis_id_t*)malloc(M * HEAP_K * sizeof(dis_id_t));
    heaps = (heap_t*)malloc(M * sizeof(heap_t));
    slots = (uint32_t*)malloc(M * (SLOTS + K) * sizeof(uint32_t));
    memset(slots, 0xff, M * SLOTS * sizeof(uint32_t));

    // heaps = new heap_t[M];
    for (uint64_t i=0; i<M; i++) {
      heaps[i].init(buf + i * HEAP_K, HEAP_K, slots + i * (SLOTS + K));
    }
  }

  ~TopQue() {

#ifdef DEBUG
    free(buf);
    free(slots);
    free(heaps);
#endif
  }


  void insert_batch_v1(float* const distances, uint32_t* ids1, uint32_t* ids2) {
    float* dis_ptr = distances;

    for (uint32_t i=0; i<BATCH; i++) {
      const uint32_t &id1 = ids1[i];
      if (id1 >= M) continue;

      heap_t &heap = heaps[id1];

      dis_ptr = distances + i * BATCH;
      for (uint32_t j=0; j<BATCH; j++, dis_ptr++) {
        const uint32_t &id2 = ids2[j];
        if (id2 >= M) continue;
        if (id1 == id2) continue;
        if (*dis_ptr > heap.heap_top) continue;

        heap.insert_unique(*dis_ptr, id2);
      }
    }
  }

  void insert_batch_v2(float* const distances, uint32_t* ids1, uint32_t* ids2) {
    float* dis_ptr = distances;
    for (uint32_t j=0; j<BATCH; j++) {
      const uint32_t &id2 = ids2[j];
      if (id2 >= M) continue;

      heap_t &heap = heaps[id2];

      dis_ptr = distances + j;
      for (uint32_t i=0; i<BATCH; i++, dis_ptr += BATCH) {
        const uint32_t &id1 = ids1[i];
        if (id1 >= M) continue;
        if (id1 == id2) continue;
        if (*dis_ptr >= heap.heap_top) continue;

        heap.insert_unique(*dis_ptr, id1);
      }
    }
  }


  void dump_output(const string output_path) {
    std::ofstream file(output_path, std::ios::binary);
    uint32_t *write_buf = (uint32_t*)malloc(K * sizeof(uint32_t));
    uint32_t id;
    uint64_t evict_cnt = 0;
    uint64_t rand_cnt = 0;

    for (int i=0; i<M; i++) {
      heap_t &heap = heaps[i];

      int j = 0;
      id = heap.pop();

      while(id != MAX_UINT32) {
        write_buf[j % K] = id;
        j++;
        id = heap.pop();
      }

#ifdef DEBUG
      // if (j < K) {
      //   cout << "Debug: warning! result not enough"
      //        << ", N: " << i
      //        << ", cnt: " << j << endl;
      // }
#endif

      while (j < K) {
        write_buf[j] = heap.pop_evicted();
        if (write_buf[j] == MAX_UINT32) {
          write_buf[j] = rand() % M;
          rand_cnt++;
        } else {
          evict_cnt++;
        }
        j++;
      }

      file.write(reinterpret_cast<char*>(write_buf), K * sizeof(uint32_t));
    }

    cout << "result evicted cnt: " << evict_cnt << " rate: " << 1.0 * evict_cnt / ((uint64_t)M * K)
         << ", rand cnt: " << rand_cnt << " rate: " << 1.0 * rand_cnt / ((uint64_t)M * K) << endl;



#ifdef DEBUG
    free(write_buf);
#endif
  }


  heap_t* heaps;
  dis_id_t* buf;
  uint32_t* slots;
};

void genRandVec(float* v) {
  for (int i = 0; i < D; i++) {
    v[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

float l2_dis(const float* v1, const float* v2) {
  double sum = 0;
  for (int i=0; i<D; i++) {
    double diff = v1[i] - v2[i];
    sum += diff * diff;
  }

  return sum;
}

float l2_dis_mm(const float* v1, const float* v2) {
  __m512 sum, va, vb, square, diff;

  sum = _mm512_setzero_ps();
  for (int i=0; i<D; i += 16) {
    va = _mm512_loadu_ps(v1 + i);
    vb = _mm512_loadu_ps(v2 + i);
    diff = _mm512_sub_ps(va, vb);
    square = _mm512_mul_ps(diff, diff);
    sum = _mm512_add_ps(sum, square);
  }

  return _mm512_reduce_add_ps(sum);
}


void run_batch(float* distances, float* v1, float* v2) {
  for (int i=0; i<BATCH; i++) {
    for (int j=0; j<BATCH; j++) {
      distances[i * BATCH + j] = l2_dis_mm(v1 + i * D, v2 + j * D);
    }
  }
}

void run_batch_mm(float* distances, const float* v1, const float* v2) {
  int i, j, k, l;
  __m512 diff, square, va, vb, reduce;
  __m512 sum[16];

  for (int i = 0; i < BATCH * D; i += 16) {
    _mm_prefetch(reinterpret_cast<const char*>(v1 + i), _MM_HINT_T0);
  }

  for (int i = 0; i < BATCH * D; i += 16) {
    _mm_prefetch(reinterpret_cast<const char*>(v2 + i), _MM_HINT_T0);
  }

  for (i = 0; i < BATCH; i++) {
    for (j = 0; j < BATCH; j += 16) {
      for (l = 0; l < 16; l++) {
        sum[l] = _mm512_setzero_ps();
      }

      for (k = 0; k < D; k += 16) {
        va = _mm512_loadu_ps(v1 + (i * D) + k);
        for (l = 0; l < 16; l++) {
          vb = _mm512_loadu_ps(v2 + (j + l) * D + k);
          diff = _mm512_sub_ps(va, vb);
          sum[l] = _mm512_fmadd_ps(diff, diff, sum[l]);
        }
      }

      for (l = 0; l < 16; l++) {
        distances[i * BATCH + j + l] = _mm512_reduce_add_ps(sum[l]);
      }
    }
  }
}

void read_from_file(string input_path) {
  std::cout << "Reading Data: " << input_path << std::endl;
  std::ifstream ifs;
  ifs.open(input_path, std::ios::binary);
  assert(ifs.is_open());

  ifs.read((char *)&M, sizeof(uint32_t));

  BATCH_M = ((M - 1) / BATCH + 1) * BATCH;

  std::cout << "Read from file with"
            << " M: " << M
            << ", K: " << K
            << ", D: " << D
            << " BATCH: " << BATCH
            << " BATCH_M: " << BATCH_M
            << ", FILE_D: " << FILE_D
            << ", Input File: " << input_path
            << std::endl;

  vectors = (float*)malloc(BATCH_M * D * sizeof(float));
  ids = (uint32_t*)malloc(BATCH_M * sizeof(uint32_t));

  float* vec_ptr = vectors;

  for (int i=0; i<M; i++) {
    ifs.read(reinterpret_cast<char*>(vec_ptr), FILE_D * sizeof(float));
    vec_ptr += FILE_D;

    if (D != FILE_D) {
      memset(vec_ptr, 0, (D - FILE_D) * sizeof(float));
      vec_ptr += D - FILE_D;
    }

    ids[i] = i;
  }

  if (BATCH_M != M) {
    memset(vec_ptr, 0, (BATCH_M - M) * D * sizeof(float));

    for (int i=M; i<BATCH_M; i++) {
      ids[i] = i;
    }
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;

  load_finished.store(true);
}

void print_stat(uint64_t batch, bool cross) {
  static atomic<size_t> batch_cnt = 0;
  static atomic<size_t> cross_cnt = 0;
  static atomic<time_t> last_time = time(nullptr);
  static atomic<size_t> last_batch = 0;
  static size_t print_interval = 100000;

  size_t cnt = batch_cnt.fetch_add(batch);
  if (cross) cross_cnt.fetch_add(batch);

  size_t new_cnt = cnt + batch;

  size_t old_value = 0;
  while (!last_batch.compare_exchange_weak(old_value, new_cnt)) {
    if (new_cnt < print_interval + old_value) {
      return;
    }
  }

  int duration = time(nullptr) - last_time;
  last_time.store(time(nullptr));
  double batch_ps = 1.0 * (new_cnt - old_value) / duration;
  double dps = batch_ps * BATCH * BATCH;
  std::cout << "\t time: " << time(nullptr) - start_time << "\tlinear: " << new_cnt - cross_cnt
            << "\tcross: " << cross_cnt
            << "\tbatch/s: " << (size_t)batch_ps << "\tdistance/s: " << dps << endl;

}

bool seed_seek_rand(uint32_t *seed_ids, float *seed_v, heap_t *seed_heap, uint32_t thread_id) {
  size_t cnt = 0;
  uint64_t id;
  static atomic<uint32_t> next_search = 0;
  static atomic<uint32_t> seed_cnt = 0;

  memset(seed_ids, 0xff, BATCH * sizeof(uint32_t));

  while (cnt < BATCH) {
    if (seed_cnt < M >> 1) {
      id = rand() % M;
      if (id >= M) continue;
    } else {
      id = next_search.fetch_add(1);
      if (id >= M) {
        if (cnt > 0) return true;
        else return false;
      }
    }

    if (topk_que->heaps[id].seed == true) {
      continue;
    }

    topk_que->heaps[id].lock();
    if (topk_que->heaps[id].seed == true) {
      topk_que->heaps[id].unlock();
      continue;
    }
    topk_que->heaps[id].seed = true;
    topk_que->heaps[id].unlock();

    seed_ids[cnt] = id;
    memcpy(seed_v + cnt * D, vectors + id * D, D * sizeof(float));
    cnt++;
    seed_cnt.fetch_add(1);
  }

  return true;
}

void seed_seek(uint32_t *seed_ids, float *v1,  heap_t *seed_heap, uint32_t thread_id) {
  static const uint64_t search_step_max = BATCH * BUILD_BATCH;
  size_t cnt = 0;
  uint32_t id = thread_id;
  while (cnt < search_step_max) {
    cnt++;
    id += THREAD;
    if (id >= M) id = thread_id;

    if (topk_que->heaps[id].seed) continue;
    seed_heap->insert_weak(1.0 * topk_que->heaps[id].insert_cnt, id);
  }

  for (int i=0; i<BATCH; i++) {
    seed_ids[i] = seed_heap->pop();
    assert(seed_ids[i] != MAX_FLOAT);
    memcpy(v1, vectors + seed_ids[i] * D, D);
    topk_que->heaps[seed_ids[i]].seed = true;
  }
}

bool linear_search(float* dis_buf, uint32_t* seed_ids, float* seed_v, heap_t *heaps) {
  for (uint32_t i=0; i<BATCH; i++) {
    heaps[i].reinit();
  }

  size_t cnt = 0;
  for (uint64_t m = 0; m < BATCH_M; m += BATCH) {
    run_batch_mm(dis_buf, seed_v, vectors + m * D);
    topk_que->insert_batch_v2(dis_buf, seed_ids, ids + m);

    float* dis_ptr;

    for (uint32_t i=0; i<BATCH; i++) {
      if (seed_ids[i] == MAX_UINT32) continue;
      heap_t &heap = heaps[i];

      dis_ptr = dis_buf + i * BATCH;
      for (uint32_t j=0; j<BATCH; j++, dis_ptr++) {
        const uint32_t id2 = m + j;
        if (id2 >= M) continue;
        if (*dis_ptr > heap.heap_top) continue;

        heap.insert_weak(*dis_ptr, id2);
      }
    }

#ifdef DEBUG
    cnt++;
    if (cnt > 1000) {
      if (time(NULL) > end_time) return false;
      print_stat(cnt, false);
      cnt = 0;
    }

#endif

  }

#ifdef DEBUG
  print_stat(cnt, false);
#else
  print_stat(BATCH_M / BATCH, false);
#endif
  return true;
}

void cross_search(float* dis_buf, uint32_t *ids1, float* v1) {
  for (int m = 0; m < BUILD_BATCH; m += BATCH) {
    for (int n = m; n < BUILD_BATCH; n += BATCH) {
      run_batch_mm(dis_buf, v1 + m * D, v1 + n * D);
      topk_que->insert_batch_v1(dis_buf, ids1 + m, ids1 + n);
      topk_que->insert_batch_v2(dis_buf, ids1 + m, ids1 + n);
    }
  }
  print_stat(((1 + BUILD_BATCH / BATCH) * (BUILD_BATCH / BATCH)) >> 1, true);
}

void build_cross_vecs(heap_t &heap, uint32_t *ids1, float* v1) {
  memset(ids1, 0xff, BUILD_BATCH * sizeof(uint32_t));
  int j = 0;

  uint32_t id = heap.pop();
  while (id != MAX_UINT32) {
    ids1[j] = id;
    memcpy(v1 + j * D, vectors + id * D, D * sizeof(float));
    j = (j + 1) % BUILD_BATCH;
    id = heap.pop();
  }
}

void knng_task (uint32_t thread_id) {
  thread_running++;
  auto round_start = std::chrono::high_resolution_clock::now();
  size_t batch_cnt = 0;
  size_t last_batch = 0;

  float* dis_buf = (float*)malloc(BATCH * BATCH * sizeof(float));
  float* v1 = (float*)malloc(BUILD_BATCH * D * sizeof(float));
  uint32_t* ids1 = (uint32_t*)malloc(BUILD_BATCH * sizeof(uint32_t));

  float* seed_v = (float*)malloc(BATCH * D * sizeof(float));
  uint32_t* seed_ids = (uint32_t*)malloc(BATCH * sizeof(uint32_t));

  dis_id_t *buf = (dis_id_t*)malloc((BATCH + 1) * LINEAR_HEAP * sizeof(dis_id_t));
  heap_t* heaps = (heap_t*)malloc((BATCH + 1) * sizeof(heap_t));

  for (uint64_t i=0; i<=BATCH; i++) {
    heaps[i].init(buf + i * LINEAR_HEAP, LINEAR_HEAP, nullptr);
  }

  heap_t *seed_heap = heaps + BATCH;
  seed_heap->max_size = BATCH;

  while(!load_finished) {
   std::this_thread::sleep_for(std::chrono::microseconds(1000));
  }

  while (time(nullptr) < end_time) {
    if (!seed_seek_rand(seed_ids, seed_v, seed_heap, thread_id)) break;
    if (!linear_search(dis_buf, seed_ids, seed_v, heaps)) break;

    for (int i=0; i<BATCH; i++) {
      if (seed_ids[i] == MAX_UINT32) continue;
      if (time(NULL) > end_time) break;

      build_cross_vecs(heaps[i], ids1, v1);

      // topk_que->heaps[seed_ids[i]].lock()
      // topk_que->heaps[seed_ids[i]].reinit();
      // topk_que->heaps[seed_ids[i]].unlock()

      cross_search(dis_buf, ids1, v1);
      topk_que->heaps[seed_ids[i]].heap_top = 0;
    }
  }

  thread_running--;

#ifdef DEBUG
  free(dis_buf);
  free(v1);
  free(ids1);

  free(seed_v);
  free(seed_ids);

  free(buf);
  free(heaps);
#endif

}

int main(int argc, char* argv[]) {
  start_time = time(nullptr);
  end_time = start_time + 28 * 60;
  srand(time(NULL));

  string input_path = "dummy-data.bin";
  string output_path = "output.bin";

  if (argc > 1) {
    input_path = string(argv[1]);
    // output_path = string(argv[2]);
  }

  load_thread = new std::thread(read_from_file, input_path);
  // read_from_file(input_path);
  while(M == 0) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
  }

  topk_que = new TopQue(M);

  std::cout << "Start Running..." << std::endl;

  vector<thread> threads;
  for (int i=0; i<THREAD; i++) {
    threads.emplace_back(knng_task, i);
  }

  while (time(nullptr) < end_time) {
   if (thread_running == 0) break;
   std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  cout << "Dumping output..." << endl;
  topk_que->dump_output(output_path);

  cout << "Freeing ... Time: " << time(nullptr) - start_time << endl;

  for (int i=0; i<THREAD; i++) {
    threads[i].join();
  }

#ifdef DEBUG
  free(vectors);
  free(ids);
  delete topk_que;
#endif

  cout << "Time use: " << time(nullptr) - start_time << endl;
  return 0;
}
