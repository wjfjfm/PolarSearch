#include <unordered_set>
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

using namespace std;

#define FILE_D 200
#define D 208
#define K 100
#define MAX_FLOAT 1e30
#define MAX_UINT32 0xffffffff
#define BATCH 256
#define MM_BATCH 4
#define LOCK_STEP 16384
#define HASH_BUCKET 130

#define LINEAR_SEED_BATCH (2 * BATCH)

#define QUE_SIZE 100
#define MAX_SEED_NUM 150000

#define THREAD 32

#define CROSS_SIZE 4096
#define COMBINE_SIZE 20

#define SEED_INACTIVE_K 4096

#define SEED_INACTIVE_DIS MAX_FLOAT

#define TIME_END 29 * 60 - 20
#define TIME_LINEAR 8 * 60

#define PACK
#define NO_FREE
// #define DEBUG

struct merge_que_t;
struct dis_id_t;
struct heap_t;
class Seed;

uint64_t M = 0;
uint64_t BATCH_M;
float* vectors;
float* vectors_ptr;
atomic_flag *locks;
uint32_t* ids;
merge_que_t* ques;
dis_id_t* buf;
heap_t *heaps;
Seed *seeds;

time_t end_time;
time_t start_time;
time_t linear_time;
std::thread* load_thread;
uint32_t lock_num;

atomic<bool> load_finished = false;
atomic<uint32_t> thread_running = 0;
atomic<uint32_t> linear_finished = 0;
atomic<uint32_t> build_finished = 0;

atomic<uint32_t> build_rela_round = 0;
atomic<uint32_t> cross_round = 0;
atomic<uint32_t> combine_round = 0;

atomic<size_t > seed_cnt = 0;
atomic<size_t > inactive_seed_cnt = 0;
atomic<size_t> linear_batch_cnt = 0;
atomic<size_t> cross_batch_cnt = 0;
atomic<size_t> combine_batch_cnt = 0;
atomic<size_t > task_cnt = 0;
atomic<size_t > task_drop_cnt = 0;
atomic<size_t> rand_cnt = 0;

uint32_t heap_size = max(CROSS_SIZE, SEED_INACTIVE_K);

class Seed {
 public:
  Seed() : id(MAX_UINT32), seed_id(MAX_UINT32) { }
  ~Seed() {
    free(cross_ids);
  }

  void insert(uint32_t id) {
    lock.lock();
    if (!exist.count(id)) {
      combine_ids.push_back(id);
    }
    lock.unlock();
  }

  uint32_t id;
  uint32_t seed_id;
  uint32_t *cross_ids;
  unordered_set<uint32_t> exist;

  mutex lock;
  vector<uint32_t> combine_ids;
};

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
  uint32_t seed_id;
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

  inline uint32_t set_seed() {
    if(!seed.test_and_set(std::memory_order_acquire)) {
      seed_id = seed_cnt.fetch_add(1);
      seeds[seed_id].id = m_id;
      seeds[seed_id].seed_id = seed_id;
      return seed_id;
    }
    return MAX_UINT32;
  }

  inline void inactive_seed() {
    if(!seed.test_and_set(std::memory_order_acquire)) {
      inactive_seed_cnt++;
    }
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
    seed_id = MAX_UINT32;
    que = _buf;
    buf = que + K;
    que2 = buf + QUE_SIZE;
    buf2 = que2 + K;

    // insert_lock.clear();
    insert_lock = new mutex();
    sort_lock.clear();
    seed.clear();

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
};

struct heap_t {
  float heap_top;
  uint32_t size;
  dis_id_t *heap;
  uint32_t max_size;
  atomic_flag insert_lock;
  unordered_set<uint32_t> *exist;

  void init(uint32_t _max_size) {
    heap = (dis_id_t*)malloc(_max_size * sizeof(dis_id_t));
    max_size = _max_size;
    exist = nullptr;
    insert_lock.clear();

    reinit();
  }

  inline void lock() {
    while(insert_lock.test_and_set()){}
  }

  inline void unlock() {
    insert_lock.clear();
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
    dis_id_t *new_heap = (dis_id_t*)malloc(new_size * sizeof(dis_id_t));
    max_size = new_size;
    free(heap);
    heap = new_heap;

    reinit();
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

void insert_batch_v1(const float* const distances, const uint32_t* ids1, const uint32_t* ids2, uint32_t size1, uint32_t size2) {
  for (uint32_t i=0; i<size1; i++) {
    const uint32_t &id1 = ids1[i];
    if (id1 >= M) continue;
    const float *dis = distances + i * BATCH;

    merge_que_t &que = ques[id1];
    for (int j=0; j<size2; j++, dis++) {
      const uint32_t &id2 = ids2[j];
      if (id2 >= M) continue;
      if (id1 == id2) continue;
      que.insert(*dis, id2);
    }
  }
}

void insert_batch_v2(const float* const distances, const uint32_t* ids1, const uint32_t* ids2, uint32_t size1, uint32_t size2) {
  for (uint32_t j=0; j<size2; j++) {
    const uint32_t &id2 = ids2[j];
    if (id2 >= M) continue;
    const float *dis = distances + j;

    merge_que_t &que = ques[id2];

    for (int i=0; i<size1; i++, dis += BATCH) {
      const uint32_t &id1 = ids1[i];
      if (id1 >= M) continue;
      if (id1 == id2) continue;
      que.insert(*dis, id1);
    }
  }
}

void insert_linear_batch(const float* const distances, const uint32_t* ids1, const uint32_t* ids2) {
  for (uint32_t j=0; j<BATCH; j++) {
    const uint32_t &id2 = ids2[j];
    if (id2 >= M) continue;
    const float *dis = distances + j;
    merge_que_t &que = ques[id2];

    for (int i=0; i<LINEAR_SEED_BATCH; i++) {
      const uint32_t &id1 = ids1[i];
      if (id1 >= M) continue;
      que.insert(*dis, id1);
      dis += BATCH;;
    }
  }
}


void dump_output_task(int fd, uint32_t from ,uint32_t to) {
  assert(from % 1024 == 0);

  const size_t buf_size = 1024;
  uint32_t *write_buf = (uint32_t*)aligned_alloc(0x400, buf_size * K * sizeof(uint32_t));
  uint32_t *buf_end = write_buf + buf_size * K;
  uint32_t *buf_ptr = write_buf;
  uint32_t  buf_first = from;

  if (to > M) to = M;

  for (int i=from; i<to; i++) {
    // ques[i].lock();
    ques[i].final_sort();
    dis_id_t *heap = ques[i].que;

    if (ques[i].que_size < K) {
      rand_cnt += K - ques[i].que_size;
    }

    for (int j=0; j<K; j++) {
      *buf_ptr = heap[j].id;
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
    free(write_buf);
#endif
}


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
    va = _mm512_load_ps(v1 + i);
    vb = _mm512_load_ps(v2 + i);
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

inline void run_linear_batch_mm(float* distances, const float* v1, const float* v2) {
  int h, i, j, k, l, m, n;
  __m512 diff;
  __m512 sum[MM_BATCH * MM_BATCH];
  __m512 va[MM_BATCH];
  __m512 vb[MM_BATCH];

  for (int i = 0; i < LINEAR_SEED_BATCH * D; i += 16) {
    _mm_prefetch(reinterpret_cast<const char*>(v1 + i), _MM_HINT_T0);
  }


  for (h = 0; h < LINEAR_SEED_BATCH; h += BATCH) {
    for (i = h * D; i < (h + BATCH) * D; i += 16) {
      _mm_prefetch(reinterpret_cast<const char*>(v2 + h), _MM_HINT_T0);
    }

    for (i = h; i < h + BATCH; i += MM_BATCH) {
      for (j = 0; j < BATCH; j += MM_BATCH) {
        for (l = 0; l < 16; l++) {
          sum[l] = _mm512_setzero_ps();
        }


        for (k = 0; k < D; k += 16) {
          for (m = 0; m<MM_BATCH; m++) {
            va[m] = _mm512_load_ps(v1 + ((i + m) * D) + k);
          }

          for (n = 0; n<MM_BATCH; n++) {
            vb[n] = _mm512_load_ps(v2 + (j + n) * D + k);
          }

          for (m = 0; m<MM_BATCH; m++) {
            for (n = 0; n<MM_BATCH; n++) {
              diff = _mm512_sub_ps(va[m], vb[n]);
              sum[m * MM_BATCH + n] = _mm512_fmadd_ps(diff, diff, sum[m * MM_BATCH + n]);
            }
          }
        }

        for (m = 0; m<MM_BATCH; m++) {
          for (n = 0; n<MM_BATCH; n++) {
            distances[(i + m) * BATCH + j + n] = _mm512_reduce_add_ps(sum[m * MM_BATCH + n]);
          }
        }
      }
    }
  }
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

  vectors_ptr = (float*)malloc(BATCH_M * D * sizeof(float) + 64);
  vectors = (float*)(((uint64_t)vectors_ptr + 64) & ~0x3f);

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

void print_stat() {
  static auto last_time = std::chrono::high_resolution_clock::now();
  static atomic<size_t> last_linear = 0;
  static atomic<size_t> last_combine = 0;
  static atomic<size_t> last_cross = 0;
  static size_t print_interval = 50000;


  auto time_now =std::chrono::high_resolution_clock::now();
  size_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_now - last_time).count();
  last_time = time_now;

  size_t linear_cnt = linear_batch_cnt;
  size_t cross_cnt = cross_batch_cnt;
  size_t combine_cnt = combine_batch_cnt;

  double linear_batch_ps = 1.0 * (linear_cnt - last_linear) * 1000 / duration;
  double cross_batch_ps = 1.0 * (cross_cnt - last_cross) * 1000 / duration;
  double combine_batch_ps = 1.0 * (combine_cnt - last_combine) * 1000 / duration;


  last_linear = linear_cnt;
  last_combine = combine_cnt;
  last_cross = cross_cnt;

  double linear_ps = linear_batch_ps * LINEAR_SEED_BATCH * BATCH;
  double cross_ps = cross_batch_ps * BATCH * BATCH;
  double combine_ps = combine_batch_ps * BATCH * BATCH;
  double dps = linear_ps + combine_ps + cross_ps;

  std::cout << std::setprecision(2)
            << "time: " << time(nullptr) - start_time
            << "\tdps: " << dps << " l(" << linear_ps << ") cr(" << cross_ps << ") cb(" << combine_ps << ")"
            << "\tbatch: l(" << last_linear << ") cr(" << last_cross << ") cb(" << last_combine << ")"
            << "\tseed: set(" << seed_cnt << ") drop(" << inactive_seed_cnt << ")"
            << "\tround: build(" << build_rela_round * LOCK_STEP << ") combine(" << combine_round << ")"
            << endl;

}


class KNNG {
  public:
  uint32_t thread_id;
  float* seed_v;
  uint32_t* seed_ids;
  heap_t* seed_heaps;
  float* dis_buf;
  float* v1;
  bool* visited;

  KNNG(uint32_t _id) : thread_id(_id){
    seed_v = (float*)aligned_alloc(64, LINEAR_SEED_BATCH * D * sizeof(float));
    seed_ids = (uint32_t*)malloc(LINEAR_SEED_BATCH * sizeof(uint32_t));
    seed_heaps = (heap_t*)malloc(LINEAR_SEED_BATCH * sizeof(heap_t));
    dis_buf = (float*)aligned_alloc(64, max(LINEAR_SEED_BATCH, BATCH) * BATCH * sizeof(float));
    visited = (bool*)malloc(lock_num * sizeof(bool));

    for (int i=0; i<LINEAR_SEED_BATCH; i++) {
      seed_heaps[i].init(heap_size);
    }
  }

  ~KNNG() {
  #ifndef NO_FREE
    free(dis_buf);

    free(seed_v);
    free(seed_ids);

    free(seed_heaps);
    free(visited);
  #endif
  }

  void seed_seek_rand() {
    size_t cnt = 0;
    uint64_t id;

    memset(seed_ids, 0xff, LINEAR_SEED_BATCH * sizeof(uint32_t));

    while (cnt < LINEAR_SEED_BATCH && seed_cnt + inactive_seed_cnt < M) {
      id = rand() % M;
      if (id >= M) continue;

      const uint32_t seed_id = ques[id].set_seed();
      if (seed_id == MAX_UINT32) {
        continue;
      }

      seed_ids[cnt] = seed_id;
      memcpy(seed_v + cnt * D, vectors + id * D, D * sizeof(float));
      cnt++;
    }
  }

  void dump_heap_to_seed() {
    for (int i=0; i<LINEAR_SEED_BATCH; i++) {
      if (seed_ids[i] == MAX_UINT32) continue;
      heap_t &heap = seed_heaps[i];
      Seed &seed = seeds[seed_ids[i]];
      seed.cross_ids = (uint32_t*)malloc(CROSS_SIZE * sizeof(uint32_t));
      seed.exist.reserve(CROSS_SIZE * 1.5);

      uint32_t j = heap_size;
      uint32_t id;
      float dis;

      while(heap.pop(&id, &dis)) {
        j--;

        if (j < SEED_INACTIVE_K && dis < SEED_INACTIVE_DIS) {
          ques[id].inactive_seed();
        }

        if (j < CROSS_SIZE) {
          seed.cross_ids[j] = id;
          seed.exist.insert(id);
        }
      }


      assert(j == 0);
      assert(seed.cross_ids[0] == seed.id);
    }
  }

  bool linear_search() {
    for (uint32_t i=0; i<LINEAR_SEED_BATCH; i++) {
      seed_heaps[i].reinit();
    }

    memset(visited, 0, lock_num * sizeof(bool));

    size_t cnt = 0;
    bool skipped = true;
    while (skipped) {
      skipped = false;
      for (int lock_id = 0; lock_id < lock_num; lock_id++) {
        if (visited[lock_id]) continue;
        if (locks[lock_id].test_and_set()) {
          skipped = true;
          continue;
        }

        visited[lock_id] = true;

        const uint32_t from = lock_id * LOCK_STEP;
        const uint32_t to =  from + LOCK_STEP > M ? M : from + LOCK_STEP;
        for (uint64_t m = from; m < to; m += BATCH) {
          run_linear_batch_mm(dis_buf, seed_v, vectors + m * D);

          for (uint32_t i=0; i<LINEAR_SEED_BATCH; i++) {
            seed_heaps[i].insert_batch(dis_buf + i * BATCH, m);
          }

          for (uint32_t i=0; i<BATCH && i + m < M; i++) {
            const uint32_t id = i + m;
            heap_t &heap = heaps[id];
            for (uint32_t j=0; j<LINEAR_SEED_BATCH; j++) {
              if (seed_ids[j] == MAX_UINT32) continue;
              heap.insert_weak(dis_buf[j * BATCH + i], seed_ids[j]);
            }
          }

          cnt++;
          if (cnt > 1000) {
            linear_batch_cnt.fetch_add(cnt);
            cnt = 0;
          }
        }

        locks[lock_id].clear();
      }
    }

    linear_batch_cnt.fetch_add(cnt);

    return true;
  }

  void build_combine_relations() {
    uint32_t lock_id = build_rela_round.fetch_add(1);

    for (; lock_id<lock_num; lock_id = build_rela_round.fetch_add(1)) {
      while(locks[lock_id].test_and_set()){}

      const uint32_t from = lock_id * LOCK_STEP;
      const uint32_t to = from + LOCK_STEP > M ? M : from + LOCK_STEP;

      for (int id = from; id < to; id++) {
        if (ques[id].seed_id != MAX_UINT32) continue;
        heap_t &heap = heaps[id];
        for (int i=0; i<COMBINE_SIZE; i++) {
          const uint32_t seed_id = heap.heap[i].id;
          if (seed_id == MAX_UINT32) continue;
          seeds[seed_id].insert(id);
        }
        // heap.resize(K);
        // heap.build_hash();
      }

      locks[lock_id].clear();
    }
  }

  void combine_search(uint32_t seed_id) {
    uint32_t *ids1 = seeds[seed_id].combine_ids.data();
    const size_t size1 = seeds[seed_id].combine_ids.size();

    uint32_t *ids2 = seeds[seed_id].cross_ids;

    size_t cnt = 0;
    for (int m = 0; m < size1; m += BATCH) {
      const size_t size = size1 - m > BATCH ? BATCH : size1 - m;

      for (int n = 0; n < CROSS_SIZE; n += BATCH) {
        run_vp_batch_mm(dis_buf, ids1 + m, ids2 + n, size, BATCH);
        insert_batch_v1(dis_buf, ids1 + m, ids2 + n, size, BATCH);
        insert_batch_v2(dis_buf, ids1 + m, ids2 + n, size, BATCH);
        cnt++;
      }
    }
    combine_batch_cnt.fetch_add(cnt);
  }

  void cross_search(uint32_t seed_id) {
    uint32_t *ids1 = seeds[seed_id].cross_ids;
    size_t cnt = 0;
    for (int m = 0; m < CROSS_SIZE; m += BATCH) {
      for (int n = m; n < CROSS_SIZE; n += BATCH) {
        run_vp_batch_mm(dis_buf, ids1 + m , ids1 + n, BATCH, BATCH);
        insert_batch_v1(dis_buf, ids1 + m, ids1 + n, BATCH, BATCH);
        insert_batch_v2(dis_buf, ids1 + m, ids1 + n, BATCH, BATCH);
        cnt++;
      }
    }
    cross_batch_cnt.fetch_add(cnt);
  }

  void linear_search_task() {
    seed_seek_rand();
    linear_search();
    dump_heap_to_seed();
  }

  bool combine_search_task() {
    uint32_t seed_id = combine_round.fetch_add(1);
    if (seed_id >= seed_cnt) return false;
    cross_search(seed_id);
    combine_search(seed_id);
    return true;
  }

  void knng_task () {
    thread_running++;
    while(!load_finished) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    while (time(NULL) < linear_time && seed_cnt + inactive_seed_cnt < M) {
      linear_search_task();
    }
    linear_finished++;

    cout << "Thread " << thread_id << " linear finihsed" << endl;

    while(linear_finished < THREAD) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    build_combine_relations();
    build_finished++;

    while(build_finished < THREAD ) {
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

#ifdef DEBUG
    if (thread_id == 0) {
      for (int i=0; i<seed_cnt; i++) {
       Seed &seed = seeds[i];
       cout << seed.combine_ids.size() << endl;
      }
    }
#endif


    while (time(NULL) < end_time) {
      if (!combine_search_task()) break;
    }

    thread_running--;
  }
};

bool compareByValue(const std::pair<int, int>& pair1, const std::pair<int, int>& pair2) {
  return pair1.second > pair2.second;
}



int main(int argc, char* argv[]) {
  assert(BATCH % MM_BATCH == 0);
  assert(LOCK_STEP % BATCH == 0);

  start_time = time(nullptr);
  end_time = start_time + TIME_END;
  linear_time = start_time + TIME_LINEAR;
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

  lock_num = ((M + LOCK_STEP - 1) / LOCK_STEP);
  locks = (atomic_flag*)malloc(lock_num * sizeof(atomic_flag));

  buf = (dis_id_t*)malloc(M * 2 * (K + QUE_SIZE) * sizeof(dis_id_t));
  ques = (merge_que_t*)malloc(M * sizeof(merge_que_t));
  heaps = (heap_t*)malloc(M * sizeof(heap_t));
  seeds = new Seed[MAX_SEED_NUM];

  for (uint64_t i=0; i<M; i++) {
    ques[i].init(buf + i * 2 * (K + QUE_SIZE), i);
  }

  for (int i=0; i<M; i++) {
    heaps[i].init(COMBINE_SIZE);
  }

  for (int i=0; i<lock_num; i++) {
    locks[i].clear();
  }

  std::cout << "Start Running..." << std::endl;

  vector<thread> threads;
  vector<KNNG*> knngs;
  for (int i=0; i<THREAD; i++) {
    knngs.emplace_back(new KNNG(i));
    threads.emplace_back(&KNNG::knng_task, knngs[i]);
  }

  print_stat();
  std::this_thread::sleep_for(std::chrono::seconds(3));

  while (time(nullptr) < end_time) {
   if (thread_running == 0) break;
   print_stat();
   std::this_thread::sleep_for(std::chrono::seconds(5));
  }

  vector<thread> dump_threads;
  cout << "Dumping output..." << endl;

  int fd = open(output_path.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
  int to = 0;
  int step = M / THREAD;
  step = max(step - step % 1024, 1024);

  for (int from = 0; from < M; from = to, to = from + step) {
    dump_threads.emplace_back(dump_output_task, fd, from, to);
  }

  for (int i=0; i<dump_threads.size(); i++) {
    dump_threads[i].join();
  }

  close(fd);

#ifndef NO_FREE
  for (int i=0; i<THREAD; i++) {
    threads[i].join();
  }
#endif

  cout << " Rand cnt: " << rand_cnt << " rate: " << 1.0 * rand_cnt / M / K << endl;

  cout << "seed cnt: " << seed_cnt << ", rate: " << 1.0 * seed_cnt / M
       << ", inactive cnt: " << inactive_seed_cnt << ", rate: " << 1.0 * inactive_seed_cnt / M
       << endl;

  cout << "Freeing ... Time: " << time(nullptr) - start_time << endl;


#ifndef NO_FREE
  free(vectors_ptr);
  free(ids);
  free(buf);
  free(ques);
  for (int i=0; i<THREAD; i++) {
    delete knngs[i];
  }
  delete[] seeds;
#endif

  cout << "Time use: " << time(nullptr) - start_time << endl;
  exit(0);
}
