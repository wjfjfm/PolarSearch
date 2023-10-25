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
#include <map>
#include <cstring>
#include <immintrin.h>
#include <math.h>

using namespace std;

#define FILE_D 200
#define D 208
#define K 100
#define MAX_QUE 10000000
#define MAX_FLOAT 1e30
#define MAX_UINT32 0xffffffff
#define BATCH 256
#define THREAD 32

uint32_t M;
uint32_t BATCH_M;
float* vectors;

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
  dis_id_t *heap;
  uint32_t size;

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

  bool insert(float dis, uint32_t id) {
    if (size == K) {
      if (dis >= heap[0].dis) return false;

      heap[0].dis = dis;
      heap[0].id = id;
      heap_down(0);
      return true;
    }

    heap[size].dis = dis;
    heap[size].id = id;
    heap_up(size);
    size++;
    return true;
  }
};

class TopQue {
 public:
  TopQue(uint32_t M) {
    buf = (dis_id_t*)malloc(M * K * sizeof(dis_id_t));
    heaps = (heap_t*)malloc(M * sizeof(heap_t));
    heap_max = (float*)malloc(M * sizeof(float));
    for (int i=0; i<M; i++) {
      heap_t &item = heaps[i];
      item.heap = buf + i * K;
      item.heap[0].dis = MAX_FLOAT;
      item.heap[0].id = MAX_UINT32;
      item.size = 1;

      heap_max[i] = MAX_FLOAT;
    }

    que = (id2_dis_t*)malloc(MAX_QUE * sizeof(id2_dis_t));
    que_head = 0;
    que_tail = 0;
    que_write = 0;
  }

  ~TopQue() {
    free(buf);
    free(heaps);
    free(heap_max);
    free(que);
  }

  void insert_batch(float* const ptr, id2_dis_t* const thread_buf, uint32_t id1, uint32_t id2) {
    float dis1;
    id2_dis_t *buf_ptr = thread_buf;
    float* dis_ptr = ptr;

    for (uint32_t i=0; i<BATCH && i+id1 < M; i++) {
      dis1 = heap_max[i + id1];
      dis_ptr = ptr + i * BATCH;
      for (uint32_t j=0; j<BATCH && j+id2 < M; j++) {

        if (*dis_ptr < dis1) {
          buf_ptr->id1 = i + id1;
          buf_ptr->id2 = j + id2;
          buf_ptr->dis = *dis_ptr;
          buf_ptr++;
        }

        if (*dis_ptr < heap_max[j + id2]) {
          buf_ptr->id1 = j + id2;
          buf_ptr->id2 = i + id1;
          buf_ptr->dis = *dis_ptr;
          buf_ptr++;
        }

        dis_ptr++;
      }
    }
    if (buf_ptr == thread_buf) return;  // nothing new

    uint64_t cnt = buf_ptr - thread_buf;

    uint64_t prev_tail = que_tail.fetch_add(cnt);
    uint64_t next_tail = prev_tail + cnt;

    while (next_tail - que_head > MAX_QUE) {
      cout << "DEBUG: top que waiting..." <<endl;
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    uint64_t copy_from = prev_tail % MAX_QUE;
    uint64_t copy_to = next_tail % MAX_QUE;

    if (copy_to > copy_from) {
      memcpy(&que[copy_from], thread_buf, cnt * sizeof(id2_dis_t));
    } else {
      memcpy(&que[copy_from], thread_buf, (MAX_QUE - copy_from) * sizeof(id2_dis_t));
      memcpy(que, thread_buf + (MAX_QUE - copy_from), copy_to * sizeof(id2_dis_t));
    }

    meta_lock.lock();
    link_buf[prev_tail] = next_tail;
    meta_lock.unlock();
  }

  // Must be single thread running
  void sort() {
    meta_lock.lock();
    while (link_buf.begin() != link_buf.end()) {
      auto ptr = link_buf.begin();
      if (link_buf.count(ptr->second)) {
        que_write = ptr->second;
        link_buf.erase(ptr);
      } else {
        break;
      }
    }
    meta_lock.unlock();

    while (que_head < que_write) {
      id2_dis_t &id2_dis = que[que_head % MAX_QUE];
      heap_t &heap = heaps[id2_dis.id1];

      if (heap.insert(id2_dis.dis, id2_dis.id2)) {
        heap_max[id2_dis.id1] = heap.heap[0].dis;
      }

      que_head++;
    }

  }

  void dump_output(const string output_path) {
    std::ofstream file(output_path, std::ios::binary);
    uint32_t *write_buf = (uint32_t*)malloc(K * sizeof(uint32_t));
    dis_id_t *heap;

    for (int i=0; i<M; i++) {
      heap = heaps[0].heap;
      for (int j=0; j<K; j++) {
        write_buf[j] = heap[j].id;
      }
      file.write(reinterpret_cast<char*>(write_buf), K * sizeof(uint32_t));
    }

    free(write_buf);
  }


  heap_t* heaps;
  dis_id_t* buf;
  float* heap_max;

  id2_dis_t* que;

  atomic<uint64_t> que_tail;
  atomic<uint64_t> que_write;
  atomic<uint64_t> que_head;

  mutex meta_lock;
  map<uint64_t, uint64_t> link_buf;
};

float calcDis(const float* v1, const float* v2) {
  int i;
  __m512 sum = _mm512_setzero_ps();
  __m512 diff, square;

  for (i = 0; i < D; i += 16) {
      // 使用 AVX-512 指令加载两个向量的数据
      __m512 va = _mm512_loadu_ps(v1 + i);
      __m512 vb = _mm512_loadu_ps(v2 + i);

      // 计算差值
      diff = _mm512_sub_ps(va, vb);

      // 计算差值的平方
      square = _mm512_mul_ps(diff, diff);

      // 累加平方值
      sum = _mm512_add_ps(sum, square);
  }

  // 将累加求和的结果水平加和
  __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum, 0), _mm512_extractf32x8_ps(sum, 1));
  __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

  // 水平加和结果向量
  float distance = _mm_cvtss_f32(_mm_hadd_ps(sum128, sum128));
  return distance;
}

// 生成一个随机向量
void genRandVec(float* v) {
  for (int i = 0; i < D; i++) {
    v[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

void run_batch(float* distances, uint32_t id1, uint32_t id2) {
    const float* vectorsA = vectors + id1 * D;
    const float* vectorsB = vectors + id2 * D;

   int i, j, k;
    __m512 sum, diff, square;

    for (i = 0; i < BATCH && i+16<=M; i += 16) {
      for (j = 0; j < BATCH && j+16<=M; j += 16) {
        sum = _mm512_setzero_ps();
        for (k = 0; k < D; k += 16) {
            __m512 va = _mm512_loadu_ps(vectorsA + (i * D) + k);
            __m512 vb;
            for (int l = 0; l < 16; l++) {
                vb = _mm512_set1_ps(vectorsB[(j * D) + k + l]);
                diff = _mm512_sub_ps(va, vb);
                square = _mm512_mul_ps(diff, diff);
                sum = _mm512_add_ps(sum, square);
            }
        }

        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum, 0), _mm512_extractf32x8_ps(sum, 1));
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

        // 存储计算得到的 L2 距离的平方
        _mm_storeu_ps(distances + (i * BATCH) + j, _mm_hadd_ps(sum128, sum128));
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

  int counter = 0;

  float* vec_ptr = vectors;

  for (int i=0; i<M; i++) {
    ifs.read(reinterpret_cast<char*>(vec_ptr), FILE_D * sizeof(float));
    vec_ptr += FILE_D;

    if (D != FILE_D) {
      memset(vectors, 0, (D - FILE_D) * sizeof(float));
      vec_ptr += D - FILE_D;
    }
  }

  if (BATCH_M != M) {
    memset(vectors, 0, D * (BATCH_M - M) * sizeof(float));
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

    std::cout << "Random gen with"
              << " M: " << M
              << ", K: " << K
              << ", D: " << D
              << " BATCH: " << BATCH
              << " BATCH_M: " << BATCH_M
              << std::endl;

    for (int i = 0; i < M; i++) {
      genRandVec(vectors + i * D);
    }

    if (BATCH_M != M) {
      memset(vectors + M * D, 0, D * (BATCH_M - M) * sizeof(float));
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

  id2_dis_t* thread_buf = (id2_dis_t*)malloc(2 * BATCH * BATCH * sizeof(id2_dis_t));
  float* dis_buf = (float*)malloc(BATCH * BATCH * sizeof(float));

  for (int n = 0; n < BATCH_M; n += BATCH) {
    for (int m = n; m < BATCH_M; m += BATCH) {
      run_batch(dis_buf, n, m);
      topk_que->insert_batch(dis_buf, thread_buf, n, m);
      topk_que->sort();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - round_start).count();
    double qps = 1.0 * BATCH * 1000 / duration;
    double dps = 1.0 * BATCH * BATCH_M * 1000 / duration;
    std::cout << "N: " << n << ", qps: " << qps << ", dps: " << dps << endl;
    round_start = std::chrono::high_resolution_clock::now();
  }

  free(thread_buf);
  free(dis_buf);

  topk_que->dump_output(output_path);

  free(vectors);
  delete topk_que;

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  int minutes = duration / 60000;
  int seconds = (duration % 60000) / 1000;
  int milliseconds = duration % 1000;

  std::cout << "Time Use:" << minutes << " m " << seconds << " s " << milliseconds << " ms" << std::endl;

  return 0;
}
