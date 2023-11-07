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
#include <cstring>
#include <immintrin.h>
#include <math.h>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <iomanip>
#include <sys/resource.h>

using namespace std;

#define FILE_D 200
#define D 208
#define K 100
#define MAX_FLOAT 1e30
#define MAX_UINT32 0xffffffff
#define BATCH 256

#define THREAD 32

#define LINEAR_SEED_BATCH 256
#define MAX_SEED_NUM 500000
#define HEAP_SIZE K
#define TARGET_INCNT 20

#define LOCK_SHIFT 16
#define LOCK_STEP (1 << LOCK_SHIFT)

#define TIME_END 1 * 60
#define TIME_LINEAR 3 * 60
#define TIME_COMBINE 25 * 60

// #define PACK
#define NO_FREE

uint64_t M = 0;

struct node_t;
class Seed;

float* vectors_ptr;
float* vectors;
uint32_t* ids;
node_t* nodes;
uint32_t* buf;
atomic_flag* locks;
Seed *seeds;

uint32_t lock_num;

time_t start_time;
time_t end_time;
time_t linear_time;
time_t combine_time;

mutex print_lock;

std::thread* load_thread;

atomic<bool> load_finished = false;
atomic<uint32_t> thread_running = 0;
atomic<uint32_t> linear_finished = 0;
atomic<uint32_t> build_finished = 0;
atomic<uint32_t> combine_finished = 0;

atomic<size_t> linear_batch_cnt = 0;
atomic<size_t> combine_batch_cnt = 0;
atomic<size_t > seed_cnt = 0;
atomic<uint32_t> build_combine_round = 0;

atomic<size_t> seek_time = 0;
atomic<size_t> search_time = 0;
atomic<size_t> run_time = 0;
atomic<size_t> insert_time = 0;
atomic<size_t> prefetch_time = 0;
atomic<size_t> load_time = 0;
atomic<size_t> sum_time = 0;
atomic<size_t> reduce_time = 0;

inline bool try_lock(uint32_t lock_id) {
  return !locks[lock_id].test_and_set(std::memory_order_acquire);
}

inline void unlock(uint32_t lock_id) {
  locks[lock_id].clear();
}

class Seed {
 public:
  std::mutex m_lock;
  uint32_t id;
  uint32_t seed_id;
  atomic<uint32_t> in_cnt;

  deque<uint32_t> cluster_ids;
  deque<uint32_t> heap_ids;

  Seed() {
    id = MAX_UINT32;
    seed_id = MAX_UINT32;
    in_cnt = 0;
  }

  void set(uint32_t _id, uint32_t _seed_id) {
    id = _id;
    seed_id = _seed_id;
  }

  inline void lock() {
    m_lock.lock();
  }

  inline void unlock() {
    m_lock.unlock();
  }

};

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
  uint32_t size;
  dis_id_t *heap;
  uint32_t max_size;

  void init(dis_id_t* _heap, uint32_t _max_size) {
    heap = _heap;
    max_size = _max_size;

    heap[0].dis = MAX_FLOAT;
    heap[0].id = MAX_UINT32;
    heap_top = MAX_FLOAT;
    size = 1;
  }

  void resize(uint32_t new_size) {
    uint32_t id;
    float dis;
    while (size > new_size) {
      pop(&id, &dis);
    }
    max_size = new_size;
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

  inline void insert(float dis, uint32_t id) {
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

struct node_t {
  heap_t heap;
  uint32_t id;
  uint32_t seed_id;

  float min_dis;
  uint32_t min_id;

  std::atomic_flag insert_lock = ATOMIC_FLAG_INIT;
  std::atomic_flag seeded = ATOMIC_FLAG_INIT;

  void init(dis_id_t* _heap, uint32_t _max_size, uint32_t _id) {
    heap.init(_heap, _max_size);

    id = _id;
    seed_id = MAX_UINT32;

    min_dis = MAX_FLOAT;
    min_id = MAX_UINT32;

    insert_lock.clear();
    seeded.clear();
  }

  inline void lock() {
    while (insert_lock.test_and_set(std::memory_order_acquire)) { }
  }

  inline void unlock() {
    insert_lock.clear(std::memory_order_release);
  }

  inline uint32_t set_seed() {
    if (!seeded.test_and_set(std::memory_order_acquire)) {
      seed_id = seed_cnt.fetch_add(1);
      seeds[seed_id].set(id, seed_id);
      return seed_id;
    }

    return MAX_UINT32;
  }

  void insert_batch(const float* dis, const uint32_t *ids, uint32_t num) {
    lock();
    for (uint32_t j=0; j<num; j++) {
      const uint32_t id2 = ids[j];
      if (id2 == id) continue;
      heap.insert(dis[j], id2);
    }
    unlock();
  }

  void insert_batch_linear(const float* dis, const uint32_t *ids, uint32_t num) {
    float _min_dis = MAX_FLOAT;
    uint32_t _min_id = MAX_UINT32;
    for (uint32_t j=0; j<num; j++) {
      const uint32_t id2 = ids[j];
      if (id2 == id) continue;

      // Perf
      heap.insert(dis[j], id2);

      if (dis[j] < _min_dis) {
        _min_dis = dis[j];
        _min_id = id2;
      }
    }

    if (_min_dis < min_dis) {
      const uint32_t new_seed_id = nodes[_min_id].seed_id;

      if (min_id != MAX_UINT32) seeds[min_id].in_cnt--;
      seeds[new_seed_id].in_cnt++;

      min_dis = _min_dis;
      min_id = _min_id;
    }
  }

};

void insert_batch_v1(const float* const distances, const uint32_t* ids1, const uint32_t* ids2, uint32_t size1, uint32_t size2) {
  for (uint32_t i=0; i<size1; i++) {
    const uint32_t &id1 = ids1[i];
    if (id1 >= M) continue;
    node_t &node = nodes[id1];
    const float *dis = distances + i * BATCH;
    node.insert_batch(dis, ids2, size2);
  }
}

// void insert_batch_v2(const float* const distances, const uint32_t* ids1, const uint32_t* ids2) {
//   for (uint32_t j=0; j<BATCH; j++) {
//     const uint32_t &id2 = ids2[j];
//     if (id2 >= M) continue;
//     const float *dis = distances + j;
//     heap_t &heap = nodes[id2].heap;
// 
//     for (int i=0; i<BATCH; i++) {
//       const uint32_t &id1 = ids1[i];
//       if (id1 >= M) continue;
//       heap.insert_unique(*dis, id1);
//       dis += BATCH;
//     }
//   }
// }

//  void insert_linear_batch(const float* const distances, const uint32_t* ids1, const uint32_t* ids2, uint32_t size) {
//    for (uint32_t j=0; j<BATCH; j++) {
//      const uint32_t &id2 = ids2[j];
//      if (id2 >= M) continue;
//      const float *dis = distances + j;
//      merge_que_t &que = ques[id2];
//
//      for (int i=0; i<size; i++) {
//        const uint32_t &id1 = ids1[i];
//        if (id1 >= M) continue;
//        que.insert(*dis, id1);
//        dis += BATCH;
//      }
//    }
//  }


void dump_output_task(int fd, uint32_t from ,uint32_t to) {
  assert(from % 1024 == 0);

  const size_t buf_size = 1024;
  uint32_t *write_ptr = (uint32_t*)malloc((buf_size * K + 0x4000) * sizeof(uint32_t));
  uint32_t *write_buf = (uint32_t*)((uint64_t)(write_ptr + 0x4000) & (uint64_t)(~0x3fff));
  uint32_t *buf_end = write_buf + buf_size * K;
  uint32_t *buf_ptr = write_buf;
  uint32_t  buf_first = from;

  if (to > M) to = M;

  for (int i=from; i<to; i++) {
    node_t &node = nodes[i];

    uint32_t id;
    float dis;
    for (int j=0; j<K; j++) {
      node.heap.pop(&id, &dis);
      *buf_ptr = id;
      buf_ptr++;
    }

    if (buf_ptr == buf_end) {
      ssize_t bytesWritten = pwrite(fd, write_buf, (buf_ptr - write_buf) * sizeof(uint32_t), buf_first * K * sizeof(uint32_t));
      assert(bytesWritten == buf_size * K * sizeof(uint32_t));
      buf_first += buf_size;
      buf_ptr = write_buf;
    }
  }

  if (buf_ptr != write_buf) {
    ssize_t bytesWritten = pwrite(fd, write_buf, (buf_ptr - write_buf) * sizeof(uint32_t), buf_first * K * sizeof(uint32_t));
    assert(bytesWritten == (buf_ptr - write_buf) * sizeof(uint32_t));
    buf_first += (buf_ptr - write_buf) / K;
    buf_ptr = write_buf;
  }

  assert(buf_first == to);

#ifndef NO_FREE
  free(write_ptr);
#endif
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


void run_vp_batch_mm(float* distances, uint32_t *idsA, uint32_t *idsB, uint32_t sizeA, uint32_t sizeB) {
  int i, j, k, l, m, n;
  __m512 diff, square;
  __m512 va;
  __m512 vb[16];
  __m512 sum[16];
  float *dis_ptr;
  __m512* sum_ptr;

  for (i = 0; i < sizeA; i++) {
    float* vpa = vectors + idsA[i] * D;
    for (int j = 0; j < D; j += 16) {
      _mm_prefetch(reinterpret_cast<const char*>(vpa + j), _MM_HINT_T0);
    }
  }

  for (i = 0; i < sizeB; i++) {
    float* vpb = vectors + idsB[i] * D;
    for (int j = 0; j < D; j += 16) {
      _mm_prefetch(reinterpret_cast<const char*>(vpb + j), _MM_HINT_T0);
    }
  }

  for (i = 0; i < sizeA; i ++) {
    for (j = 0; j < sizeB; j += 16) {
      for (l = 0; l < 16; l++) {
        sum[l] = _mm512_setzero_ps();
      }

      for (k = 0; k < D; k += 16) {
        sum_ptr = sum;
        va = _mm512_load_ps(vectors + idsA[i] * D + k);

        for (n = 0; n < 16; n++) {
          vb[n] = _mm512_load_ps(vectors + idsB[n + j] * D + k);
        }

        for (n = 0; n < 16; n++) {
          diff = _mm512_sub_ps(va, vb[n]);
          *sum_ptr = _mm512_fmadd_ps(diff, diff, *sum_ptr);
          sum_ptr++;
        }
      }

      sum_ptr = sum;
      dis_ptr = distances + i * BATCH + j;
      for (n = 0; n < 16; n++) {
        *dis_ptr = _mm512_reduce_add_ps(*sum_ptr);
        dis_ptr++;
        sum_ptr++;
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

  std::cout << "Read from file with"
            << " M: " << M
            << ", K: " << K
            << ", D: " << D
            << " BATCH: " << BATCH
            << ", FILE_D: " << FILE_D
            << ", Input File: " << input_path
            << std::endl;

  vectors_ptr = (float*)malloc(M * D * sizeof(float) + 64);
  vectors = (float*)(((uint64_t)vectors_ptr + 64) & ~0x3f);
  ids = (uint32_t*)malloc(M * sizeof(uint32_t));

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


  ifs.close();
  std::cout << "Finish Reading Data" << endl;

  load_finished.store(true);
}

void print_setting() {
  cout << "Settings: "
       << " THREAD " << THREAD << endl
       << " TIME_END " << TIME_END << endl;
}
void print_stat() {
  static auto last_time = std::chrono::high_resolution_clock::now();
  static atomic<size_t> last_linear = 0;
  static atomic<size_t> last_combine = 0;

  auto time_now =std::chrono::high_resolution_clock::now();
  size_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_now - last_time).count();
  last_time = time_now;

  size_t linear_cnt = linear_batch_cnt;
  size_t combine_cnt = combine_batch_cnt;

  double linear_batch_ps = 1.0 * (linear_cnt - last_linear) * 1000 / duration / BATCH;
  double combine_batch_ps = 1.0 * (combine_cnt - last_combine) * 1000 / duration;


  last_linear = linear_cnt;
  last_combine = combine_cnt;

  double linear_ps = linear_batch_ps * BATCH * BATCH;
  double combine_ps = combine_batch_ps * BATCH * BATCH;
  double dps = linear_ps + combine_ps;

  float dps_balance = 1.0 * linear_cnt / (linear_cnt + combine_cnt * BATCH);

  std::cout << std::setprecision(2)
            << "time: " << time(nullptr) - start_time
            << "\tdps: " << dps << " l(" << linear_ps << ") c(" << combine_ps << ")"
            << "\tbps: l(" << (size_t)linear_batch_ps  << ") c(" << (size_t)combine_batch_ps << ")"
            << "\tseed: set(" << seed_cnt << ") build(" << build_combine_round << ")"
            << endl;
}


class KNNG {
 public:
  uint32_t* seed_ids;
  uint32_t* seed_seed_ids;
  float* dis_buf;
  bool* visited;
  uint32_t thread_id;

  KNNG(uint32_t _thread_id) {
    thread_id = _thread_id;
    seed_ids = (uint32_t*)malloc(BATCH * sizeof(uint32_t));
    dis_buf = (float*)malloc(BATCH * BATCH * sizeof(float));
    visited = (bool*)malloc(lock_num * sizeof(bool));
  }

  ~KNNG() {
#ifndef NO_FREE
    free(dis_buf);
    free(seed_ids);
    free(visited);
#endif
  }

  void linear_search() {
    size_t cnt = 0;

    for (int i=0; i<lock_num; i++) {
      visited[i] = false;
    }

    bool skipped = true;
    while (skipped) {
      skipped = false;
      for (uint64_t lock_id = 0; lock_id < lock_num ; lock_id++) {
        if (visited[lock_id]) continue;
        if (!try_lock(lock_id)) {
          skipped = true;
          continue;
        }

        visited[lock_id] = true;

        for (uint64_t m = lock_id << LOCK_SHIFT; (m < (lock_id + 1) << LOCK_SHIFT) && m < M; m += BATCH) {
          const uint32_t size_v = m + BATCH > M ? M - m : BATCH;
          static auto start_time = std::chrono::high_resolution_clock::now();

          run_vp_batch_mm(dis_buf, ids + m, seed_ids, size_v, LINEAR_SEED_BATCH);

          static auto mid_time = std::chrono::high_resolution_clock::now();

          for (int i=0; i<size_v; i++) {
            const uint32_t id = m + i;
            assert(id < M);
            const float *dis = dis_buf + i * BATCH;
            nodes[id].insert_batch_linear(dis, seed_ids, LINEAR_SEED_BATCH);
          }

          static auto end_time = std::chrono::high_resolution_clock::now();

          cnt++;
          if (cnt > 1000) {
            linear_batch_cnt.fetch_add(cnt * LINEAR_SEED_BATCH);
            cnt = 0;
          }
        }

        unlock(lock_id);
      }
    }

    linear_batch_cnt.fetch_add(cnt * LINEAR_SEED_BATCH);
  }

  void seed_seek_rand() {
    size_t cnt = 0;
    uint64_t id;

    while (cnt < LINEAR_SEED_BATCH) {
      id = rand() % M;

      const uint32_t min_id = nodes[id].min_id;
      if (min_id != MAX_UINT32) {
        if (seeds[min_id].in_cnt < TARGET_INCNT) continue;
      }

      uint32_t seed_id = nodes[id].set_seed();
      if (seed_id == MAX_UINT32) continue;

      seed_ids[cnt] = id;
      cnt++;
    }
  }

  void build_combine_relations() {

    uint32_t id =  build_combine_round.fetch_add(1);
    while (id < M) {
      node_t &node = nodes[id];
      for (int i=0; i<HEAP_SIZE; i++) {
        const uint32_t seed_id = node.heap.heap[i].id;
        assert(seed_id < M);
        Seed &seed = seeds[nodes[seed_id].seed_id];
        seed.lock();
        seed.heap_ids.push_back(id);
        seed.unlock();
      }

      const uint32_t seed_id = node.min_id;
      assert(seed_id < M);
      Seed &seed = seeds[nodes[seed_id].seed_id];
      seed.lock();
      seed.cluster_ids.push_back(id);
      seed.unlock();

      id =  build_combine_round.fetch_add(1);
    }
  }

  void knng_task () {
    thread_running++;


    while(!load_finished) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    while (time(NULL) < linear_time) {
      seed_seek_rand();
      linear_search();
    }

    linear_finished++;
    cout << "Thread " << thread_id << " Linear Search Finished" << endl;

    while(linear_finished < THREAD) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }


    build_combine_relations();

    build_finished++;
    cout << "Thread " << thread_id << " Relation Build Finished" << endl;

    while(build_finished < THREAD) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }



    thread_running--;
  }

};

int main(int argc, char* argv[]) {
  start_time = time(nullptr);
  end_time = start_time + TIME_END;
  linear_time = start_time + TIME_LINEAR;
  combine_time = start_time + TIME_COMBINE;

  srand(time(NULL));

  // struct rlimit core_limit;
  // core_limit.rlim_cur = 0;
  // core_limit.rlim_max = 0;
  // if (setrlimit(RLIMIT_CORE, &core_limit) != 0) {
  //     std::cerr << "Failed to set core file limit" << std::endl;
  //     exit(1);
  // }

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

  dis_id_t *bufs = (dis_id_t*)malloc(M * max(K, HEAP_SIZE)  * sizeof(dis_id_t));
  nodes = (node_t*)malloc(M * sizeof(node_t));
  lock_num = (M + LOCK_STEP - 1) >> LOCK_SHIFT;
  locks = (atomic_flag*)malloc(lock_num * sizeof(atomic_flag));
  seeds = new Seed[MAX_SEED_NUM];

  for (uint64_t i=0; i<M; i++) {
    nodes[i].init(bufs + i * max(K, HEAP_SIZE), HEAP_SIZE, i);
  }

  for (uint64_t i=0; i<lock_num; i++) {
    locks[i].clear();
  }

  std::cout << "Start Running..." << std::endl;

  vector<thread> threads;
  KNNG* knngs[THREAD];

  for (int i=0; i<THREAD; i++) {
    knngs[i] = new KNNG(i);
    threads.emplace_back(&KNNG::knng_task, knngs[i]);
  }

  print_setting();
  print_stat();
  std::this_thread::sleep_for(std::chrono::seconds(3));

  while (time(nullptr) < end_time) {
   if (thread_running == 0) break;
   print_stat();

   std::this_thread::sleep_for(std::chrono::seconds(3));
  }

  vector<thread> dump_threads;
  cout << "Dumping output..." << endl;

  int fd = open(output_path.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
  int to = 0;
  int step = M / THREAD;
  step = step - step % 1024;

  for (int from = 0; from < M; from = to, to = from + step) {
    dump_threads.emplace_back(&dump_output_task, fd, from, to);
  }

  for (int i=0; i<dump_threads.size(); i++) {
    dump_threads[i].join();
  }

  close(fd);

#ifndef NO_FREE
  for (int i=0; i<THREAD; i++) {
    threads[i].join();
    delete knngs[i];
  }

#endif

  cout << "Freeing ... Time: " << time(nullptr) - start_time << endl;


#ifndef NO_FREE
  free(vectors_ptr);
  free(ids);
  free(bufs);
  free(nodes);
  free(locks);
  delete[] seeds;
#endif

  cout << "Time use: " << time(nullptr) - start_time << endl;
  exit(0);
}
