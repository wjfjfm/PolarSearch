#include <csignal>
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
#include <unordered_set>

using namespace std;

#define FILE_D 200
#define D 208
#define K 100
#define MAX_FLOAT 1e30
#define MAX_UINT32 0xffffffff
#define BATCH 256

#define THREAD 32

#define LINEAR_SEED_BATCH (1 * BATCH)
#define LINEAR_VECT_BATCH (32 * BATCH)

#define MAX_SEED_NUM (10 * LINEAR_SEED_BATCH)
#define HEAP_SIZE 100
#define HASH_BUCKET 130

#define HEAP_CUT_THD (40 * BATCH)
#define TARGET_INCNT 30000
#define SEED_SEEK_DEPTH 100000

#define TIME_END 30 * 60

//#define PACK
#define NO_FREE
// #define DEBUG

uint64_t M = 0;
uint64_t BATCH_M = 0;

struct node_t;
class Seed;
struct task_t;

float* vectors_ptr;
float* vectors;
uint32_t* ids;
node_t* nodes;
uint32_t* buf;
Seed *seeds;
vector<task_t> tasks;
atomic_flag *locks;

atomic<uint64_t> linear_vec_batch = 0;
atomic<uint32_t> build_combine_round = 0;
atomic<uint32_t> gen_task_round = 0;
atomic<uint32_t> combine_search_round = 0;

time_t start_time;
time_t end_time;

mutex print_lock;
bool printed = false;

std::thread* load_thread;

atomic<bool> load_finished = false;
atomic<uint32_t> thread_running = 0;
atomic<uint32_t> linear_finished = 0;
atomic<uint32_t> build_finished = 0;
atomic<uint32_t> task_finished = 0;

atomic<size_t> linear_batch_cnt = 0;
atomic<size_t> combine_batch_cnt = 0;
atomic<size_t > seed_cnt = 0;
atomic<size_t > task_cnt = 0;
double avg_incnt = 0;

uint32_t max_incnt_id = MAX_UINT32;
uint32_t max_incnt = 0;

// 信号处理函数
void signalHandler(int signal) {
  static int cnt = 0;
  std::cout << "Received signal: " << signal << std::endl;
  end_time = 0;
  cnt++;

  if (cnt > 3) exit(0);
}

struct task_t {
  uint32_t *clus_ids;
  uint32_t size_c;
  uint32_t *heap_ids;
  uint32_t size_h;
};

class Seed {
 public:
  std::mutex m_lock;
  uint32_t id;
  uint32_t seed_id;

  vector<uint32_t> cluster_ids;
  vector<uint32_t> heap_ids;

  Seed() {
    id = MAX_UINT32;
    seed_id = MAX_UINT32;
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
  unordered_set<uint32_t> *exist;

  void init(uint32_t _max_size) {
    heap = (dis_id_t*)malloc(_max_size * sizeof(dis_id_t));
    max_size = _max_size;
    exist = nullptr;

    reinit();
  }

  void reinit() {
    heap[0].dis = MAX_FLOAT;
    heap[0].id = MAX_UINT32;
    heap_top = MAX_FLOAT;
    size = 1;

    if (exist) { exist->clear(); }
  }

  void build_hash() {
    assert(exist == nullptr);
    exist = new unordered_set<uint32_t>(HASH_BUCKET);
    for (int i=0; i<size; i++) {
      const auto ret = exist->insert(heap[i].id);
      assert(ret.second);
    }
  }

  void resize(uint32_t new_size) {
    assert(!exist);
    if (exist) delete exist, exist = nullptr;
    uint32_t id;
    float dis;
    while (size > new_size) {
      pop(&id, &dis);
    }
    heap_top = heap[0].dis;
    max_size = new_size;

    dis_id_t *new_heap = (dis_id_t*)malloc(new_size * sizeof(dis_id_t));
    memcpy(new_heap, heap, size * sizeof(dis_id_t));
    free(heap);

    heap = new_heap;
  }

  void uninit() {
    free(heap);
    if (exist) delete(exist), exist = nullptr;
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

  inline void insert_unique(float dis, uint32_t id) {
    assert(exist);
    if (size == max_size) {
      if (dis >= heap_top) { return; }

      const auto ret = exist->insert(id);
      if (!ret.second) return;
      exist->erase(heap[0].id);

      heap[0].dis = dis;
      heap[0].id = id;
      heap_down(0);

      heap_top = heap[0].dis;
      return;
    }

    const auto ret = exist->insert(id);
    if (!ret.second) return;

    heap[size].dis = dis;
    heap[size].id = id;
    heap_up(size);
    size++;
  }

  inline void insert_weak(float dis, uint32_t id) {
    assert(exist == nullptr);
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

  atomic<uint32_t> in_cnt;
  unordered_set<uint32_t> *exist;
  std::atomic_flag insert_lock = ATOMIC_FLAG_INIT;
  std::atomic_flag seeded = ATOMIC_FLAG_INIT;

  void init(uint32_t _max_size, uint32_t _id) {
    heap.init(_max_size);

    id = _id;
    seed_id = MAX_UINT32;

    min_dis = MAX_FLOAT;
    min_id = MAX_UINT32;

    exist = nullptr;
    in_cnt = 0;

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

      if (seed_id >= MAX_SEED_NUM) {
        seed_cnt.store(MAX_SEED_NUM);
        seeded.clear();
        return MAX_UINT32;
      }

      seeds[seed_id].set(id, seed_id);
      return seed_id;
    }

    return MAX_UINT32;
  }

  void insert_batch_v1(const float* dis, const uint32_t *ids, uint32_t num) {
    lock();
    for (uint32_t j=0; j<num; j++) {
      const uint32_t id2 = ids[j];
      if (id2 == id) continue;
      heap.insert_weak(dis[j], id2);
    }
    unlock();
  }

  void insert_batch_v2(const float* dis, const uint32_t *ids, uint32_t num) {
    lock();
    for (uint32_t j=0; j<num; j++) {
      const uint32_t id2 = ids[j];
      if (id2 == id) continue;
      heap.insert_unique(dis[j * BATCH], id2);
    }
    unlock();
  }

  void insert_batch_linear(const float* dis_buf, const uint32_t *ids, uint32_t num) {
    if (seed_id != MAX_UINT32) return;
    float _min_dis = MAX_FLOAT;
    uint32_t _min_id = MAX_UINT32;
    for (uint32_t j=0; j<num; j++) {
      const uint32_t id2 = ids[j];
      const float dis = dis_buf[j * BATCH];
      assert(id2 != id);

      heap.insert_weak(dis, id2);

      if (dis < min_dis && dis < _min_dis) {
        _min_dis = dis;
        _min_id = id2;
      }
    }

    if (_min_dis < min_dis) {
      if (min_id != MAX_UINT32) nodes[min_id].in_cnt--;
      nodes[_min_id].in_cnt++;
      min_dis = _min_dis;
      min_id = _min_id;
    }
  }

};

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


void run_vp_batch_mm(float* distances, uint32_t* const idsA, uint32_t* const idsB, uint32_t sizeA, uint32_t sizeB) {
  int i, j, k, l, m, n;
  __m512 diff, square;
  __m512 va[4];
  __m512 vb[4];
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

  for (i = 0; i < sizeA; i += 4) {
    for (j = 0; j < sizeB; j += 4) {
      for (l = 0; l < 16; l++) {
        sum[l] = _mm512_setzero_ps();
      }

      for (k = 0; k < D; k += 16) {
        for (m = 0; m < 4 && m + i < sizeA; m++) {
          va[m] = _mm512_load_ps(vectors + idsA[m + i] * D + k);
        }

        for (n = 0; n < 4 && n + j < sizeB; n++) {
          vb[n] = _mm512_load_ps(vectors + idsB[n + j] * D + k);
        }

        sum_ptr = sum;
        for (m = 0; m < 4; m++) {
          for (n = 0; n < 4; n++) {
            diff = _mm512_sub_ps(va[m], vb[n]);
            *sum_ptr = _mm512_fmadd_ps(diff, diff, *sum_ptr);
            sum_ptr++;
          }
        }
      }

      for (m = 0; m < 4; m++) {
        for (n = 0; n < 4; n++) {
          distances[(i + m) * BATCH + j + n] = _mm512_reduce_add_ps(sum[m * 4 + n]);
        }
      }
    }
  }
}

void run_batch_mm(float* distances, const float* v1, const float* v2) {
  int i, j, k, l, m, n;
  __m512 diff, square;
  __m512 va;
  __m512 vb[16];
  __m512 sum[256];

  for (int i = 0; i < BATCH * D; i += 16) {
    _mm_prefetch(reinterpret_cast<const char*>(v1 + i), _MM_HINT_T0);
  }

  for (int i = 0; i < BATCH * D; i += 16) {
    _mm_prefetch(reinterpret_cast<const char*>(v2 + i), _MM_HINT_T0);
  }

  for (i = 0; i < BATCH; i += 16) {
    for (j = 0; j < BATCH; j += 16) {
      for (l = 0; l < 256; l++) {
        sum[l] = _mm512_setzero_ps();
      }

      for (k = 0; k < D; k += 16) {
        for (n = 0; n < 16; n++) {
          vb[n] = _mm512_loadu_ps(v2 + (j + n) * D + k);
        }

        l = 0;
        for (m = 0; m < 16; m++) {
          va = _mm512_loadu_ps(v1 + (i + m) * D + k);
          for (n = 0; n < 16; n++) {
            diff = _mm512_sub_ps(va, vb[n]);
            sum[l] = _mm512_fmadd_ps(diff, diff, sum[l]);
            l++;
          }
        }
      }

      l = 0;
      for (m = 0; m < 16; m++) {
        for (n = 0; n < 16; n++) {
          distances[(i + m) * BATCH + j + n] = _mm512_reduce_add_ps(sum[l]);
          l++;
        }
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
       << " LINEAR_SEED_BATCH " << LINEAR_SEED_BATCH << endl
       << " LINEAR_VECT_BATCH " << LINEAR_VECT_BATCH << endl
       << " MAX_SEED_NUM " << MAX_SEED_NUM << endl;
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

  double linear_batch_ps = 1.0 * (linear_cnt - last_linear) * 1000 / duration;
  double combine_batch_ps = 1.0 * (combine_cnt - last_combine) * 1000 / duration;


  last_linear = linear_cnt;
  last_combine = combine_cnt;

  double linear_ps = linear_batch_ps * BATCH * BATCH;
  double combine_ps = combine_batch_ps * BATCH * BATCH;
  double dps = linear_ps + combine_ps;

  float dps_balance = 1.0 * linear_cnt / (linear_cnt + combine_cnt * BATCH);

  uint64_t total_incnt = 0;

  for (int i=0; i<seed_cnt; i++) {
    Seed &seed = seeds[i];
    uint64_t incnt = nodes[seed.id].in_cnt;
    total_incnt += incnt * incnt;

    if (incnt > max_incnt || seed.id == max_incnt_id) {
      max_incnt = incnt;
      max_incnt_id = seed.id;
    }
  }

  avg_incnt = 1.0 * total_incnt / M;

  std::cout << std::setprecision(2)
            << "time: " << time(nullptr) - start_time
            << "\tdps: " << dps << " l(" << linear_ps << ") c(" << combine_ps << ")"
            << "\tbatch: l(" << linear_cnt  << ") c(" << combine_cnt << ") l/c(" << 1.0 * linear_cnt / combine_cnt << ")"
            << "\tseed: set(" << seed_cnt << ")"
            << "\tround: linear(" << linear_vec_batch << ") build(" << build_combine_round << ")"
            << "\ttask: cnt(" << task_cnt << ") round(" << combine_search_round << ")"
            << "\tincnt avg(" << avg_incnt << ") max(" << max_incnt << ") max_id(" << max_incnt_id << ")"
            << endl;
}

void seed_seek_rand(uint32_t target_seed_cnt) {
  uint32_t max_id = MAX_UINT32;
  uint32_t max_incnt = 0;
  uint32_t seek_depth = 0;

  while (seed_cnt < target_seed_cnt) {
    uint32_t id = rand() % M;

    if (nodes[id].min_id != MAX_UINT32) {
      if (nodes[nodes[id].min_id].in_cnt < avg_incnt) continue;
    }

    nodes[id].set_seed();
  }
}

bool compareTask(const task_t &A, const task_t &B) {
  return (uint64_t)A.size_h * A.size_c < (uint64_t)B.size_h * B.size_c;
}

void build_combine_relations() {
  uint32_t id = build_combine_round.fetch_add(1);
  for (; id < M; id = build_combine_round.fetch_add(1)) {
    node_t &node = nodes[id];
    if (node.seed_id != MAX_UINT32) continue;

    for (int i=0; i<HEAP_SIZE; i++) {
      const uint32_t node_id = node.heap.heap[i].id;
      assert(node_id != MAX_UINT32);
      Seed &seed = seeds[nodes[node_id].seed_id];
      seed.lock();
      seed.heap_ids.push_back(id);
      seed.unlock();
    }

    Seed &seed = seeds[nodes[node.min_id].seed_id];
    seed.lock();
    seed.cluster_ids.push_back(id);
    seed.unlock();

    node.heap.resize(K);
    // node.heap.build_hash();
  }
}

class KNNG {
 public:
  float* dis_buf;
  float* dis_ptr;
  uint32_t thread_id;
  uint32_t* seed_ids;
  heap_t* heaps;

  KNNG(uint32_t _thread_id) {
    thread_id = _thread_id;
    dis_ptr = (float*)malloc(LINEAR_SEED_BATCH * BATCH * sizeof(float) + 64);
    dis_buf = (float*)(((uint64_t)dis_ptr + 64) & ~0x3f);
    seed_ids = (uint32_t*)malloc(LINEAR_SEED_BATCH * sizeof(uint32_t));
    heaps = (heap_t*)malloc(LINEAR_SEED_BATCH * sizeof(heap_t));

    for (int i=0; i<LINEAR_SEED_BATCH; i++) {
      heaps[i].init(K);
    }
  }

  ~KNNG() {
#ifndef NO_FREE
    for (int i=0; i<LINEAR_SEED_BATCH; i++) {
      heaps[i].uninit();
    }

    free(dis_ptr);
    free(seed_ids);
    free(heaps);
#endif
  }

  void load_seed_ids(uint32_t from_seed_id) {
    seed_seek_rand(from_seed_id + LINEAR_SEED_BATCH);
    for (int i=0; i<LINEAR_SEED_BATCH; i++) {
      while (seeds[from_seed_id + i].id == MAX_UINT32) {
        cout << "Debug load seed MAX" << endl;
      }
      seed_ids[i] = seeds[from_seed_id + i].id;
    }
  }

  void combine_search() {
    uint32_t task_id = combine_search_round.fetch_add(1);
    for (;task_id < task_cnt; task_id = combine_search_round.fetch_add(1)) {
      task_t &task = tasks[task_id];
      const uint32_t size_H = task.size_h;
      const uint32_t size_C = task.size_c;

      for (int h=0; h<size_H; h+=BATCH) {
        uint32_t size_h = size_H - h >= BATCH ? BATCH : size_H - h;

        for (int c=0; c<size_C; c+=BATCH) {
          uint32_t size_c = size_C - c >= BATCH ? BATCH : size_C - c;

          run_vp_batch_mm(dis_buf, task.heap_ids + h, task.clus_ids + c, size_h, size_c);
          combine_batch_cnt++;

          for(int i=0; i<size_h; i++) {
            node_t &node = nodes[task.heap_ids[i + h]];
            node.insert_batch_v1(dis_buf + i * BATCH, task.clus_ids + c, size_c);
          }

          // for(int i=0; i<size_c; i++) {
          //   node_t &node = nodes[task.clus_ids[i + c]];
          //   node.insert_batch_v2(dis_buf + i, task.heap_ids + h, size_h);
          // }
        }
      }
    }
  }

  void linear_search() {
    uint32_t last_round = MAX_UINT32;
    uint64_t vec_batch = linear_vec_batch.fetch_add(LINEAR_VECT_BATCH);

    while (true) {
      uint32_t new_round = vec_batch / BATCH_M;

      uint32_t m_from = vec_batch % BATCH_M;


      if (new_round != last_round) {
        if (last_round != MAX_UINT32) {
          for (int i=0; i<LINEAR_SEED_BATCH; i++) {
            node_t &node = nodes[seed_ids[i]];
            dis_id_t *heap = heaps[i].heap;
            node.lock();

            if (node.min_id != node.id) {
              if (node.min_id != MAX_UINT32) {
                nodes[node.min_id].in_cnt--;
              }
              node.min_id = node.id;
              node.min_dis = 0;
              node.heap.reinit();
              node.heap.resize(K);
            }

            for (int j=0; j<K; j++) {
              node.heap.insert_weak(heap[j].dis, heap[j].id);
            }

            node.unlock();
            heaps[i].reinit();
          }
        }

        if (new_round * LINEAR_SEED_BATCH >=  MAX_SEED_NUM) break;

        load_seed_ids(new_round * LINEAR_SEED_BATCH);
        last_round = new_round;
        for (int i=0; i<LINEAR_SEED_BATCH; i++) {
          for (int j=0; j<D; j+=16) {
            _mm_prefetch(reinterpret_cast<const char*>(vectors + seed_ids[i] * D + j), _MM_HINT_T1);
          }
        }
      }

      for (int i=0; i<LINEAR_VECT_BATCH; i++) {
        for (int j=0; j<D; j+=16) {
          _mm_prefetch(reinterpret_cast<const char*>(vectors + (i + m_from) * D + j), _MM_HINT_T1);
        }
      }

     #ifndef PACK
           while (locks[m_from / LINEAR_VECT_BATCH].test_and_set()) {}
     #endif

      for (uint32_t m = m_from; (m < m_from + LINEAR_VECT_BATCH) && m < M; m += BATCH) {
        const uint32_t size_v = m + BATCH > M ? M - m : BATCH;

        for (int n=0; n<LINEAR_SEED_BATCH; n += BATCH) {
          run_vp_batch_mm(dis_buf + n * BATCH, seed_ids + n, ids + m, BATCH, size_v);
        }

        linear_batch_cnt.fetch_add(LINEAR_SEED_BATCH / BATCH);

        for (int i=0; i<size_v; i++) {
          const uint32_t id = m + i;
          nodes[id].insert_batch_linear(dis_buf + i, seed_ids, LINEAR_SEED_BATCH);
        }

        for (int i=0; i<LINEAR_SEED_BATCH; i++) {
          for (int j=0; j<size_v; j++) {
            if (seed_ids[i] == m + j) continue;
            heaps[i].insert_weak(dis_buf[i * BATCH + j], m + j);
          }
        }

      }

#ifndef PACK
      locks[m_from / LINEAR_VECT_BATCH].clear();
#endif

      vec_batch = linear_vec_batch.fetch_add(LINEAR_VECT_BATCH);
    }
  }

  void gen_task() {
    static atomic_flag choose = ATOMIC_FLAG_INIT;
    if (choose.test_and_set()) return;

    uint32_t seed_id = gen_task_round.fetch_add(1);
    for (; seed_id < seed_cnt; seed_id = gen_task_round.fetch_add(1)) {
      Seed &seed = seeds[seed_id];
      if (seed.cluster_ids.size() == 0) continue;
      if (seed.heap_ids.size() == 0) continue;

      for (int i=0; i<seed.heap_ids.size(); i+=HEAP_CUT_THD) {
        uint32_t size = seed.heap_ids.size() - i > HEAP_CUT_THD ? HEAP_CUT_THD : seed.heap_ids.size() - i;


        uint32_t task_id = task_cnt.fetch_add(1);
        tasks.emplace_back();
        task_t &task = tasks[task_id];
        task.clus_ids = seed.cluster_ids.data();
        task.size_c = seed.cluster_ids.size();

        task.heap_ids = seed.heap_ids.data() + i;
        task.size_h = size;
      }
    }

    sort(tasks.data(), tasks.data() + task_cnt, compareTask);

#ifdef DEBUG
    for (int i=0; i<task_cnt; i++) {
      cout << tasks[i].size_h << "x" << tasks[i].size_c << " ";
    }
#endif
  }


  void knng_task () {
    thread_running++;

    while(!load_finished) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    linear_search();
    linear_finished++;
    cout << "Thread " << thread_id << " linear finished" << endl;

    while(linear_finished < THREAD) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    assert(seed_cnt == MAX_SEED_NUM);

    build_combine_relations();
    build_finished++;
    cout << "Thread " << thread_id << " build finished" << endl;


    while(build_finished < THREAD) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    // print_lock.lock();
    // if (!printed) {
    //   uint32_t sum_clus = 0;
    //   uint32_t sum_heap = 0;

    //   for (int i=0; i<MAX_SEED_NUM; i++) {
    //     Seed &seed = seeds[i];
    //     cout << "seed id: " << seed.seed_id
    //          << ", id: " << seed.id
    //          << ", cluster_size: " << seed.cluster_ids.size()
    //          << ", heap_size: " << seed.heap_ids.size()
    //          << endl;

    //     bool exit = false;
    //     cout << "Clus ids: " << endl;
    //     for (int j=0; j< seed.cluster_ids.size(); j++) {
    //       if (seed.cluster_ids[j] == seed.id) exit = true;
    //       cout << *(&seed.cluster_ids[0] + j) << " ";
    //     }

    //     cout << endl << "Heap ids: " << endl;
    //     for (int j=0; j< seed.heap_ids.size(); j++) {
    //       cout << seed.heap_ids[j] << " ";
    //     }
    //     cout << endl << endl;

    //     sum_clus += seed.cluster_ids.size();
    //     sum_heap += seed.heap_ids.size();

    //   }
    //   cout << "Sum clus: " << sum_clus << " sum heap: " << sum_heap << endl;
    //   printed = true;
    // }
    // print_lock.unlock();

    gen_task();
    task_finished++;

    cout << "Thread " << thread_id << " gen task finished" << endl;


    while(task_finished < THREAD) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    combine_search();

    cout << "Thread " << thread_id << " finished" << endl;

    thread_running--;
  }

};

int main(int argc, char* argv[]) {

  signal(SIGINT, signalHandler);

  start_time = time(nullptr);
  end_time = start_time + TIME_END;

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

  BATCH_M = M % LINEAR_VECT_BATCH == 0 ? M : M + LINEAR_VECT_BATCH - (M % LINEAR_VECT_BATCH);
  cout << "M: " << M << " BATCH_M: " << BATCH_M << endl;

#ifdef PACK
  if (M == 10000) {
    int fd = open(output_path.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
    close(fd);
    return 0;
  }
#endif

  nodes = (node_t*)malloc(M * sizeof(node_t));
  seeds = new Seed[MAX_SEED_NUM];
  locks = (atomic_flag*)malloc((BATCH_M / LINEAR_VECT_BATCH) * sizeof(atomic_flag));

  for (uint64_t i=0; i<M; i++) {
    nodes[i].init(HEAP_SIZE, i);
  }

  for (int i=0; i< BATCH_M / LINEAR_VECT_BATCH; i++) {
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
  if (step == 0) step = 1024;

  cout << "step: " << step << endl;

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
  for (int i=0; i<M; i++) {
    nodes[i].heap.uninit();
  }

  free(vectors_ptr);
  free(ids);
  free(nodes);
  free(locks);
  delete[] seeds;
#endif

  cout << "Time use: " << time(nullptr) - start_time << endl;
  exit(0);
}
