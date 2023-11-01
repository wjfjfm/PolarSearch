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
#include <fcntl.h>
#include <unistd.h>

using namespace std;

#define FILE_D 200
#define D 208
#define K 100
#define MAX_FLOAT 1e30
#define MAX_UINT32 0xffffffff
#define BATCH 256
#define THREAD 40
#define LINEAR_THREAD 20

#define TASK_SET_SIZE 10000
#define QUE_SIZE 200

#define COMBINE_SIZE 8192

#define CROSS_SIZE 2048

#define SEED_INACTIVE_K 2048
#define SEED_INACTIVE_DIS MAX_FLOAT

#define TIME_LIMIT 27 * 60

#define DEBUG
#define PACK

uint64_t M = 0;
uint64_t BATCH_M;
float* vectors;
uint32_t* ids;
time_t end_time;
time_t start_time;
std::thread* load_thread;

atomic<bool> load_finished = false;
atomic<uint32_t> thread_running = 0;
atomic<uint32_t> thread_linear = 0;
atomic<uint32_t> thread_combine = 0;

atomic<size_t > seed_cnt = 0;
atomic<size_t > inactive_seed_cnt = 0;
atomic<size_t > combine_cnt = 0;
atomic<size_t> rand_cnt = 0;

atomic<uint32_t> search_round1 = 0;
atomic<uint32_t> search_round2 = 0;

class TopQue;
TopQue *topk_que;

struct dis_id_t {
  float dis;
  uint32_t id;

};

inline bool compare_dis_id(const dis_id_t& A, const dis_id_t& B) {
  return A.dis < B.dis || A.dis == B.dis && A.id < B.id;
}

struct id2_dis_t {
  uint32_t id1;
  uint32_t id2;
  float dis;
};

struct merge_que_t {
  float hwm;
  uint32_t m_id;
  uint32_t buf_size;
  uint32_t que_size;
  dis_id_t *que;
  dis_id_t *buf;

  dis_id_t *que2;
  dis_id_t *buf2;

  // std::atomic_flag insert_lock = ATOMIC_FLAG_INIT;
  std::mutex *insert_lock;
  std::atomic_flag sort_lock = ATOMIC_FLAG_INIT;
  std::atomic_flag seed = ATOMIC_FLAG_INIT;
  std::atomic_flag combine = ATOMIC_FLAG_INIT;

  inline bool set_seed() {
    if(!seed.test_and_set(std::memory_order_acquire)) {
      seed_cnt++;
      return true;
    }
    return false;
  }

  inline void inactive_seed() {
    if(!seed.test_and_set(std::memory_order_acquire)) {
      inactive_seed_cnt++;
    }
  }

  inline bool set_combine() {
    if (!combine.test_and_set(std::memory_order_acquire)) {
      combine_cnt++;
      return true;
    }
    return false;
  }
  inline void disable_combine() {
    combine.test_and_set(std::memory_order_acquire);
  }

  inline void lock() {
    // while (insert_lock.test_and_set(std::memory_order_acquire)) { }
    insert_lock->lock();
  }

  inline void unlock() {
    // insert_lock.clear(std::memory_order_release);
    insert_lock->unlock();
  }

  inline void sort_set() {
    while (sort_lock.test_and_set(std::memory_order_acquire)) { }
  }
  inline void sort_done() {
    sort_lock.clear(std::memory_order_release);
  }

  void init(dis_id_t* _buf, uint32_t _id) {
    m_id = _id;
    que = _buf;
    buf = que + K;
    que2 = buf + QUE_SIZE;
    buf2 = que2 + K;

    // insert_lock.clear();
    insert_lock = new mutex();
    sort_lock.clear();
    seed.clear();
    combine.clear();

    hwm = MAX_FLOAT;
    que_size = 0;
    buf_size = 0;
  }


  void final_sort() {
    sort_set();
    uint32_t size = buf_size;
    buf_size = 0;
    swap(buf, buf2);

    merge_sort(size);

    sort_done();
  }

  inline void merge_sort(uint32_t size) {
    sort(buf2, buf2 + size, compare_dis_id);

    swap(que, que2);
    int i=0, j=0, k=0;
    uint32_t last_insert = MAX_UINT32;
    while (k < K) {
      if (i == que_size && j == size) break;
      if (j == size || (i != que_size && compare_dis_id(que2[i], buf2[j]))) {
        if (que2[i].id != last_insert) {
          que[k] = que2[i];
          last_insert = que[k].id;
          k++;
          i++;
        } else {
          i++;
        }
      } else{
        if (buf2[j].id != last_insert) {
          que[k] = buf2[j];
          last_insert = buf2[j].id;
          k++;
          j++;
        } else {
          j++;
        }
      }
    }

    que_size = k;
    if (que_size == K) {
      hwm = que[K-1].dis;
    }
  }

  void insert(float dis, uint32_t id) {
    if (dis >= hwm) return;
    if (id == m_id) return;
    lock();
    buf[buf_size].dis = dis;
    buf[buf_size].id = id;
    buf_size++;

    if (buf_size != QUE_SIZE) {
      unlock();
      return;
    }

    sort_set();
    buf_size = 0;
    swap(buf, buf2);
    unlock();

    merge_sort(QUE_SIZE);
    sort_done();
    return;
  }

  void insert_batch_2(float *dis, uint32_t *ids) {
    for (int i=0; i<BATCH; i++) {
      if (ids[i] >= M) continue;
      insert(dis[i * BATCH], ids[i]);
    }
  }

  void insert_batch(float *dis, uint32_t *ids) {
    for (int i=0; i<BATCH; i++) {
      if (ids[i] >= M) continue;
      insert(dis[i], ids[i]);
    }
  }

};

struct heap_t {
  float heap_top;
  uint32_t size;
  dis_id_t *heap;
  uint32_t max_size;


  void init(dis_id_t* _heap, uint32_t _max_size) {
    heap = _heap;
    max_size = _max_size;

    reinit();
  }

  void reinit() {
    heap[0].dis = MAX_FLOAT;
    heap[0].id = MAX_UINT32;
    heap_top = MAX_FLOAT;
    size = 1;
  }

  inline void heap_down(uint32_t index) {
    uint32_t left = 2 * index + 1;
    uint32_t right = 2 * index + 2;
    uint32_t largest = index;

    while (true) {
      if (left < size && heap[left].dis > heap[largest].dis) {
        largest = left;
      }

      if (right < size && heap[right].dis > heap[largest].dis) {
        largest = right;
      }

      if (largest == index) {
        break;
      }

      std::swap(heap[index], heap[largest]);
      index = largest;
      left = 2 * index + 1;
      right = 2 * index + 2;
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

  void insert_batch(float *dis, uint32_t id) {
    for (int i=0; i<BATCH && i + id < M; i++) {
      if (size == max_size) {
        if (dis[i] >= heap_top) { continue; }

        heap[0].dis = dis[i];
        heap[0].id = id + i;
        heap_down(0);

        heap_top = heap[0].dis;
        continue;
      }

      heap[size].dis = dis[i];
      heap[size].id = id + i;
      heap_up(size);
      size++;
    }
  }

  bool pop(uint32_t *id, float *dis) {
    if (size == 0) return false;

    *id = heap[0].id;
    *dis = heap[0].dis;

    size--;

    if (size > 0) {
      heap[0].id = heap[size].id;
      heap[0].dis = heap[size].dis;
      heap_down(0);
    }

    return true;
  }

};

class TopQue {
 public:
  TopQue(uint64_t M) {
    buf = (dis_id_t*)malloc(M * 2 * (K + QUE_SIZE) * sizeof(dis_id_t));
    // memset(buf, 0xff, M * 2 * (K + QUE_SIZE) * sizeof(dis_id_t));
    ques = (merge_que_t*)malloc(M * sizeof(merge_que_t));

    for (uint64_t i=0; i<M; i++) {
      ques[i].init(buf + i * 2 * (K + QUE_SIZE), i);
    }

  }

  ~TopQue() {

#ifdef DEBUG
    free(buf);
    free(ques);
#endif
  }


  void insert_batch_v1(float* const distances, uint32_t* ids1, uint32_t* ids2) {
    float* dis_ptr = distances;

    for (uint32_t i=0; i<BATCH; i++) {
      const uint32_t &id1 = ids1[i];
      if (id1 >= M) continue;
      ques[id1].insert_batch(distances + i * BATCH, ids2);
    }
  }

  void insert_batch_v2(float* const distances, uint32_t* ids1, uint32_t* ids2) {
    float* dis_ptr = distances;
    for (uint32_t j=0; j<BATCH; j++) {
      const uint32_t &id2 = ids2[j];
      if (id2 >= M) continue;
      ques[id2].insert_batch_2(distances + j, ids1);
    }
  }


  void dump_output_task(int fd, uint32_t from ,uint32_t to) {

    uint32_t *write_buf = (uint32_t*)malloc(K * sizeof(uint32_t));

    if (to > M) to = M;

    for (int i=from; i<to; i++) {
      // ques[i].lock();
      ques[i].final_sort();
      dis_id_t *heap = ques[i].que;

      if (ques[i].que_size < K) {
        rand_cnt += K - ques[i].que_size;
      }

      for (int j=0; j<K; j++) {
        write_buf[j] = heap[j].id;
      }
      //ques[i].unlock();

      ssize_t bytesWritten = pwrite(fd, write_buf, K * sizeof(uint32_t), i * K * sizeof(uint32_t));
      assert(bytesWritten == K * sizeof(uint32_t));
    }
#ifdef DEBUG
    free(write_buf);
#endif
  }

  merge_que_t* ques;
  dis_id_t* buf;

  std::mutex lock;
  std::deque<uint32_t*> tasks;
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

void print_stat(uint64_t batch) {
  static atomic<size_t> batch_cnt = 0;
  static auto last_time = std::chrono::high_resolution_clock::now();
  static atomic<size_t> last_batch = 0;
  static size_t print_interval = 50000;

  size_t cnt = batch_cnt.fetch_add(batch);

  size_t new_cnt = cnt + batch;

  size_t old_value = 0;
  while (!last_batch.compare_exchange_weak(old_value, new_cnt)) {
    if (new_cnt < print_interval + old_value) {
      return;
    }
  }

  auto time_now =std::chrono::high_resolution_clock::now();
  size_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_now - last_time).count();
  last_time = time_now;
  double batch_ps = 1.0 * (new_cnt - old_value) * 1000 / duration;
  double dps = batch_ps * BATCH * BATCH;
  std::cout << "time: " << time(nullptr) - start_time << "\tbatch: " << new_cnt
            << "\tseed_cnt: " << seed_cnt << " inactive: " << inactive_seed_cnt << " combine: " << combine_cnt
            << "\tseed seek round1: " << search_round1 << " round2: " << search_round2
            << "\tthread linear: " << thread_linear << " combine: " << thread_combine
            << "\tbatch/s: " << (size_t)batch_ps << "\tdistance/s: " << dps << endl;

}

uint32_t seed_seek_rand(uint32_t *seed_ids, float *seed_v) {
  size_t cnt = 0;
  uint64_t id;

  memset(seed_ids, 0xff, BATCH * sizeof(uint32_t));

  while (cnt < BATCH) {
    if (inactive_seed_cnt < M >> 1) {
      id = rand() % M;
      if (id >= M) continue;
      if (topk_que->ques[id].que_size == K) continue;
    } else if (search_round1 < M) {
      id = search_round1.fetch_add(1);
      if (id >= M) continue;
      if (topk_que->ques[id].que_size == K) continue;
    } else if (inactive_seed_cnt < M - 100000) {
      id = rand() % M;
      if (id >= M) continue;
    } else if (search_round2 < M) {
      id = search_round2.fetch_add(1);
      if (id >= M) continue;
    } else {
      return cnt;
    }

    if (!topk_que->ques[id].set_seed()) {
      continue;
    }

    seed_ids[cnt] = id;
    memcpy(seed_v + cnt * D, vectors + id * D, D * sizeof(float));
    cnt++;
  }

  return cnt;
}

bool linear_search(float *dis_buf, const uint32_t* seed_ids, const float* seed_v, heap_t *heaps) {
  for (uint32_t i=0; i<BATCH; i++) {
    heaps[i].reinit();
  }

  size_t cnt = 0;
  for (uint64_t m = 0; m < BATCH_M; m += BATCH) {
    run_batch_mm(dis_buf, seed_v, vectors + m * D);

    for (uint32_t i=0; i<BATCH; i++) {
      if (seed_ids[i] == MAX_UINT32) continue;
      heaps[i].insert_batch(dis_buf + i * BATCH, m);
    }

    cnt++;
    if (cnt > 1000) {
      if (time(NULL) >= end_time) return false;
      print_stat(cnt);
      cnt = 0;

    }
  }

  print_stat(cnt);

  return true;
}

void combine_search(float* dis_buf, uint32_t *ids1, float* v1, uint32_t size1, uint32_t *ids2, float *v2, uint32_t size2) {

  size_t cnt = 0;
  for (int m = 0; m < size1; m += BATCH) {
    for (int n = 0; n < size2; n += BATCH) {
      run_batch_mm(dis_buf, v1 + m * D, v2 + n * D);
      topk_que->insert_batch_v1(dis_buf, ids1 + m, ids2 + n);
      topk_que->insert_batch_v2(dis_buf, ids1 + m, ids2 + n);
      cnt++;
    }
  }
  print_stat(cnt);
}

void cross_search(float* dis_buf, uint32_t *ids1, float* v1) {
  size_t cnt = 0;
  for (int m = 0; m < CROSS_SIZE; m += BATCH) {
    for (int n = m; n < CROSS_SIZE; n += BATCH) {
      run_batch_mm(dis_buf, v1 + m * D, v1 + n * D);
      topk_que->insert_batch_v1(dis_buf, ids1 + m, ids1 + n);
      topk_que->insert_batch_v2(dis_buf, ids1 + m, ids1 + n);
      cnt++;
    }
  }
  print_stat(cnt);
}

void linear_search_task(float* dis_buf, heap_t* heaps, float* seed_v, uint32_t *seed_ids) {
  uint32_t seek_num = seed_seek_rand(seed_ids, seed_v);
  if (seek_num == 0) return;

  if (!linear_search(dis_buf, seed_ids, seed_v, heaps)) return;

  for (int i=0; i<BATCH; i++) {
    if (seed_ids[i] == MAX_UINT32) break;

    uint32_t* set_ids = (uint32_t*)malloc((CROSS_SIZE + COMBINE_SIZE) * sizeof(uint32_t));

    heap_t &heap = heaps[i];

    uint32_t id;
    float dis;

    int j = CROSS_SIZE + COMBINE_SIZE;
    while(heap.pop(&id, &dis)) {
      j--;

      if (j < CROSS_SIZE + COMBINE_SIZE) {
        set_ids[j] = id;
      }

      if (j < SEED_INACTIVE_K && dis <= SEED_INACTIVE_DIS) {
        topk_que->ques[id].inactive_seed();
        // topk_que->ques[id].disable_combine();
      }

    }

    assert(set_ids[0] == seed_ids[i]);
    assert(j == 0);

    topk_que->lock.lock();
    topk_que->tasks.push_back(set_ids);
    topk_que->lock.unlock();
  }
}

bool compareByValue(const std::pair<int, int>& pair1, const std::pair<int, int>& pair2) {
  return pair1.second > pair2.second;
}

void combine_search_task(float* dis_buf, uint32_t* ids1, float* v1, uint32_t* ids2, float* v2) {

  for (int i=0; i<CROSS_SIZE; i++) {
    memcpy(v1 + i * D, vectors + ids1[i] * D, D * sizeof(float));
  }

  for (int i=0; i<COMBINE_SIZE; i++) {
    memcpy(v2 + i * D, vectors + ids2[i] * D, D * sizeof(float));
  }

  cross_search(dis_buf, ids1, v1);
  combine_search(dis_buf, ids1, v1, CROSS_SIZE, ids2, v2, COMBINE_SIZE);
}

void knng_task (uint32_t thread_id) {
  thread_running++;

  float* seed_v = (float*)malloc(BATCH * D * sizeof(float));
  uint32_t* seed_ids = (uint32_t*)malloc(BATCH * sizeof(uint32_t));

  dis_id_t *buf = (dis_id_t*)malloc((BATCH) * (CROSS_SIZE + COMBINE_SIZE) * sizeof(dis_id_t));
  heap_t* heaps = (heap_t*)malloc((BATCH) * sizeof(heap_t));

  float* dis_buf = (float*)malloc(BATCH * BATCH * sizeof(float));
  float* v1 = (float*)malloc(CROSS_SIZE * D * sizeof(float));
  float* v2 = (float*)malloc(COMBINE_SIZE * D * sizeof(float));

  for (uint64_t i=0; i<BATCH; i++) {
    heaps[i].init(buf + i * (CROSS_SIZE + COMBINE_SIZE) , CROSS_SIZE + COMBINE_SIZE);
  }

  while(!load_finished) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
  }

  while (time(NULL) < end_time) {
    bool linear_task = false;
    if (thread_id < LINEAR_THREAD) linear_task = true;
    uint32_t *set_ids;

    topk_que->lock.lock();
    const uint32_t task_set_size = topk_que->tasks.size();

    if (task_set_size == 0) linear_task = true;
    if (task_set_size >= TASK_SET_SIZE) linear_task = false;

    if (!linear_task) {
      set_ids = topk_que->tasks.back();
      topk_que->tasks.pop_back();
    }

    topk_que->lock.unlock();

    if (linear_task) {
      thread_linear++;
      linear_search_task(dis_buf, heaps, seed_v, seed_ids);
      thread_linear--;
    } else {
      if (!topk_que->ques[set_ids[0]].set_combine()) {
        free(set_ids);
        continue;
      }

      thread_combine++;

      uint32_t *ids1 = set_ids;
      uint32_t *ids2 = set_ids + CROSS_SIZE;

      combine_search_task(dis_buf, ids1, v1, ids2, v2);

      free(set_ids);

      thread_combine--;
    }
  }


  #ifdef DEBUG
    free(dis_buf);
    free(v1);
    free(v2);

    free(seed_v);
    free(seed_ids);

    free(buf);
    free(heaps);
  #endif

  thread_running--;
}

int main(int argc, char* argv[]) {
  start_time = time(nullptr);
  end_time = start_time + TIME_LIMIT;
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

#ifdef PACK
  if (M == 10000) {
    int fd = open(output_path.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
    close(fd);
    return 0;
  }
#endif

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

  vector<thread> dump_threads;
  cout << "Dumping output..." << endl;

  int fd = open(output_path.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
  int to = 0;
  for (int from = 0; from < M; from = to) {
    to = from + (M - 1) / THREAD + 1;
    dump_threads.emplace_back(&TopQue::dump_output_task, topk_que, fd, from, to);
  }
  // topk_que->dump_output(output_path);

  for (int i=0; i<dump_threads.size(); i++) {
    dump_threads[i].join();
  }
  close(fd);

  // for (int i=0; i<THREAD; i++) {
  //   threads[i].join();
  // }

  cout << " Rand cnt: " << rand_cnt << " rate: " << 1.0 * rand_cnt / M / K << endl;

  cout << "seed cnt: " << seed_cnt << ", rate: " << 1.0 * seed_cnt / M
       << ", inactive cnt: " << inactive_seed_cnt << ", rate: " << 1.0 * inactive_seed_cnt / M
       << endl;

  cout << "Freeing ... Time: " << time(nullptr) - start_time << endl;


#ifdef DEBUG
  free(vectors);
  free(ids);
  delete topk_que;
#endif

  cout << "Time use: " << time(nullptr) - start_time << endl;
  return 0;
}
