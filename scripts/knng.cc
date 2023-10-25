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

using namespace std;

#define FILE_D 200
#define D 208
#define K 100
#define MAX_QUE 0x8000000
#define MAX_FLOAT 1e30
#define MAX_UINT32 0xffffffff
#define BATCH 256
#define THREAD 32

uint32_t M;
uint32_t BATCH_M;
float* vectors;
uint32_t* ids;

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

class heap_t {
 public:
  float heap_top;

  dis_id_t *heap;
  uint32_t size;
  uint32_t max_size;
  unordered_set<uint32_t> edges;
  std::mutex lock;

  void init(dis_id_t* _heap, uint32_t _max_size) {
    heap = _heap;
    heap[0].dis = MAX_FLOAT;
    heap[0].id = MAX_UINT32;
    heap_top = MAX_FLOAT;
    size = 1;
    max_size = _max_size;
    edges.reserve(max_size);
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

  void insert(float dis, uint32_t id) {
    if (size == max_size) {
      if (dis >= heap[0].dis) return;

      auto result = edges.insert(id);
      if (result.second == false) return;

      edges.erase(heap[0].id);
      heap[0].dis = dis;
      heap[0].id = id;
      heap_down(0);

      heap_top = heap[0].dis;
      return;
    }

    auto result = edges.insert(id);
    if (result.second == false) return;

    heap[size].dis = dis;
    heap[size].id = id;
    heap_up(size);
    size++;

    // no need to replace heap_top because it is always MAX_FLOAT
    // heap_top = dis;
  }
};

class TopQue {
 public:
  TopQue(uint32_t M) {
    buf = (dis_id_t*)malloc(M * K * sizeof(dis_id_t));
    heaps = new heap_t[M];
    for (int i=0; i<M; i++) {
      heaps[i].init(buf + i * K, K);
    }

  }

  ~TopQue() {
    free(buf);
    delete[] heaps;
  }


  void insert_batch(float* const distances, uint32_t* ids1, uint32_t* ids2) {
    float* dis_ptr = distances;

    for (dis_ptr = distances; dis_ptr < distances + BATCH * BATCH; dis_ptr += 16) {
      _mm_prefetch(reinterpret_cast<const char*>(dis_ptr), _MM_HINT_T0);
    }

    for (int i = 0; i < BATCH; i += 16) {
      _mm_prefetch(reinterpret_cast<const char*>(ids1 + i), _MM_HINT_T0);
    }

    for (int i = 0; i < BATCH; i += 16) {
      _mm_prefetch(reinterpret_cast<const char*>(ids2 + i), _MM_HINT_T0);
    }

    bool hold_lock = false;
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

        if (!hold_lock) {
          heap.lock.lock();
          hold_lock = true;
        }

        heap.insert(*dis_ptr, id2);
      }

      if (hold_lock) heap.lock.unlock();
    }

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

        if (!hold_lock) {
          heap.lock.lock();
          hold_lock = true;
        }

        heap.insert(*dis_ptr, id1);
      }

      if (hold_lock) heap.lock.unlock();
    }

  }


  void dump_output(const string output_path) {
    // kassert(que_head == que_tail);
    // kassert(que_write == que_tail);
    std::ofstream file(output_path, std::ios::binary);
    uint32_t *write_buf = (uint32_t*)malloc(K * sizeof(uint32_t));
    dis_id_t *heap;

    for (int i=0; i<M; i++) {
      heap = heaps[i].heap;
      for (int j=0; j<K; j++) {
        assert(heap[j].id != MAX_UINT32);
        write_buf[j] = heap[j].id;
      }
      file.write(reinterpret_cast<char*>(write_buf), K * sizeof(uint32_t));
    }

    free(write_buf);
  }


  heap_t* heaps;
  dis_id_t* buf;
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

  int counter = 0;

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
    memset(vec_ptr, 0, D * (BATCH_M - M) * sizeof(float));

    for (int i=M; i<BATCH_M; i++) {
      ids[i] = i;
    }
  }

  ifs.close();
  std::cout << "Finish Reading Data" << endl;
}
int main(int argc, char* argv[]) {

  string input_path = "input.bin";
  string output_path = "output.bin";

  if (argc == 2) {
    M = std::atoi(argv[1]);
    BATCH_M = ((M - 1) / BATCH + 1) * BATCH;

    vectors = (float*)malloc(BATCH_M * D * sizeof(float));
    ids = (uint32_t*)malloc(BATCH_M * sizeof(uint32_t));

    std::cout << "Random gen with"
              << " M: " << M
              << ", K: " << K
              << ", D: " << D
              << " BATCH: " << BATCH
              << " BATCH_M: " << BATCH_M
              << std::endl;

    for (int i = 0; i < M; i++) {
      genRandVec(vectors + i * D);
      ids[i] = i;
    }

    if (BATCH_M != M) {
      memset(vectors + M * D, 0, D * (BATCH_M - M) * sizeof(float));

      for (int i=M; i<BATCH_M; i++) {
        ids[i] = i;
      }
    }
  } else {
    if (argc == 3) {
      input_path = string(argv[1]);
      output_path = string(argv[2]);
    }

    read_from_file(input_path);
  }

  topk_que = new TopQue(M);

  std::cout << "Start Running..." << std::endl;
  auto round_start = std::chrono::high_resolution_clock::now();
  auto start = std::chrono::high_resolution_clock::now();

  float* dis_buf = (float*)malloc(BATCH * BATCH * sizeof(float));

  for (int n = 0; n < BATCH_M; n += BATCH) {
    for (int m = n; m < BATCH_M; m += BATCH) {
      run_batch_mm(dis_buf, vectors + n * D, vectors + m * D);
      topk_que->insert_batch(dis_buf, ids + n, ids + m);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - round_start).count();
    double qps = 1.0 * BATCH * 1000 / duration;
    double dps = 1.0 * BATCH * BATCH_M * 1000 / duration;
    std::cout << "N: " << n << ", qps: " << qps << ", dps: " << dps << endl;
    round_start = std::chrono::high_resolution_clock::now();
  }

  free(dis_buf);

  topk_que->dump_output(output_path);

  free(vectors);
  free(ids);
  delete topk_que;

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  int minutes = duration / 60000;
  int seconds = (duration % 60000) / 1000;
  int milliseconds = duration % 1000;

  std::cout << "Time Use:" << minutes << " m " << seconds << " s " << milliseconds << " ms" << std::endl;

  return 0;
}
