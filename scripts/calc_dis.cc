#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>
#include <immintrin.h>
#include <thread>
#include <atomic>
#include <cstring>
#include <unordered_set>
#include <mutex>
#include <vector>

#define PRINT_DETAIL
#define APPRO_NUM 1000
#define MAX_FLOAT 1e30
#define MAX_UINT32 0xffffffff

using namespace std;

uint32_t M;
uint32_t K = 100;
uint32_t D = 208;
uint32_t FILE_D = 200;
uint32_t THREAD = 32;

float* vectors;
uint32_t* outputs;

std::mutex print_lock;
atomic<size_t> total_hit = 0;
atomic<size_t> total_num = 0;
atomic<size_t> all_hit = 0;

atomic<size_t> accurate_hit = 0;
atomic<size_t> accurate_num = 0;


void load_knng(string path, uint32_t *knng) {
  std::ifstream ifs;
  ifs.open(path, std::ios::binary);
  uint32_t* vec_ptr = knng;
  for (int i=0; i<M; i++) {
    ifs.read(reinterpret_cast<char*>(vec_ptr), K * sizeof(uint32_t));
    sort(vec_ptr, vec_ptr + K);
    vec_ptr += K;
  }
}

struct dis_id_t {
  float dis;
  uint32_t id;
};

class Heap {
  public:
  float heap_top;
  uint32_t size;
  dis_id_t *heap;
  uint32_t max_size;

  Heap(uint32_t _max_size) {
    max_size = _max_size;
    heap = (dis_id_t*)malloc(max_size * sizeof(dis_id_t));
    heap[0].dis = MAX_FLOAT;
    heap[0].id = MAX_UINT32;
    heap_top = MAX_FLOAT;
    size = 1;
  }

  ~Heap() {
    free(heap);
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

  void insert(float dis, uint32_t id) {
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



void calc_recall(uint32_t* outputs, uint32_t* answers) {
  size_t cnt = 0;

  for (int i=0; i<M; i++) {
    uint32_t *p1 = outputs + i * K;
    uint32_t *p2 = answers + i * K;
    uint32_t* const end_p1 = p1 + K;
    uint32_t* const end_p2 = p2 + K;

    while (p1 < end_p1 && p2 < end_p2) {
      if (*p1 == *p2) {
        cnt++;
        p1++;
        p2++;
      } else if (*p1 < *p2){
        p1++;
      } else {
        p2++;
      }
    }
  }

  size_t total = (uint64_t)M * (uint64_t)K;
  double recall = 1.0 * cnt / total;
  cout << "recall: " << recall
       << ", total: " << total
       << ", hit: " << cnt
       << ", miss: " << total - cnt
       << endl << endl;
}

double l2_dis(float* v1, float* v2) {
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

void calc_dis(float* vectors, uint32_t* knng) {
  double sum = 0;

  for (int i=0; i<M; i++) {
    float* v1 = vectors + i * D;
    for (int j=0; j<K; j++) {
      float* v2 = vectors + knng[i * K + j] * D;
      sum += l2_dis(v1, v2);
    }
  }

  double avg = sum / M / K;

  cout << setprecision(20) << "Total l2 dis: " << sum
       << ", avg dis: " << avg
       << endl << endl;
}

void calc_dis_task(float* vectors, uint32_t* knng, uint32_t from, uint32_t to, double *result) {
  double sum = 0;

  for (int i=from; i<to && i<M; i++) {
    float* v1 = vectors + i * D;
    for (int j=0; j<K; j++) {
      float* v2 = vectors + knng[i * K + j] * D;
      sum += l2_dis_mm(v1, v2);
    }
  }

  *result = sum;
}

void calc_dis_mm(float* vectors, uint32_t* knng) {
  vector<thread> threads;
  uint32_t batch = M / THREAD;
  vector<double> results(THREAD, 0);

  double sum = 0;

  for (int i=0; i<THREAD; i++) {
    threads.emplace_back(calc_dis_task, vectors, knng, (M / THREAD) * i, (M / THREAD) * (i + 1), &results[i]);
  }

  for (int i=0; i<THREAD; i++) {
    threads[i].join();
    sum += results[i];
  }

  double avg = sum / M / K;

  cout << setprecision(20) << "Total l2 dis: " << sum
       << ", avg dis: " << avg
       << endl << endl;
}

uint64_t calc_recall_appro(float* vectors, uint32_t id, uint32_t *output) {
  Heap heap(K);
  for (int i=0; i<M; i++) {
    if (i == id) continue;
    float dis = l2_dis_mm(vectors + id * D, vectors + i * D);
    heap.insert(dis, i);
  }

  unordered_set<uint32_t> exist;
  for(int i=0; i<K; i++) {
    exist.insert(output[i]);
  }

#ifdef PRINT_DETAIL
  int pos = K;
  char* write_buf = (char*) malloc((K+1) * sizeof(char));
  write_buf[pos--] = '\0';
#else
  int pos = K - 1;
#endif

  int hit = 0;
  uint32_t rid;
  float dis;

#ifdef PRINT_DETAIL
  float max_dis, min_dis;
#endif


  for(int i=0; i<K; i++) {
    heap.pop(&rid, &dis);

    if (exist.count(rid)) {
      hit++;

#ifdef PRINT_DETAIL
      write_buf[pos] = '+';
#endif

    } else {

#ifdef PRINT_DETAIL
      write_buf[pos] = '-';
#endif
    }

#ifdef PRINT_DETAIL
    if (pos == K-1) max_dis = dis;
    else if (pos == 0) min_dis = dis;

#endif
    pos--;
  }

  total_hit.fetch_add(hit);
  total_num++;

  if (hit == 100) all_hit++;

#ifdef PRINT_DETAIL

  string color = "";

  if (hit < 20) {
    color = "\033[31m";
  } else if (hit < 50) {
    color = "\033[33m";
  } else if (hit < 80) {
    color = "\033[36m";
  }

  if (hit >= 20) {
    accurate_hit += hit;
    accurate_num++;
  }

  float accurate_recall = 1.0 * accurate_hit / accurate_num / K;
  float recall = 1.0 * total_hit / total_num / K;

  print_lock.lock();
  fprintf(stdout, "recall:%0.4f acc_recal:%0.4f %sid:%7i hit:%3i min:%06.3f max:%06.3f:%s\033[0m\n", recall, accurate_recall, color.c_str(), id, hit, min_dis, max_dis, write_buf);
  print_lock.unlock();
#endif

  return hit;
}

void recall_appro_task(size_t num) {
  for (int i=0; i<num; i++) {
    uint32_t rand_id = rand() % M;
    calc_recall_appro(vectors, rand_id, outputs + rand_id * K);
  }
}

int main(int argc, char* argv[]) {

  srand(time(NULL));

  string output_path = "output.bin";
  string answer_path = "result.bin";
  string vector_path = "dummy-data.bin";

  bool with_answer = true;

  if (argc >= 3) {
    vector_path = string(argv[1]);
    output_path = string(argv[2]);
    with_answer = false;
  }

  if (argc == 4) {
    answer_path = string(argv[3]);
    with_answer = true;
  }

  std::ifstream ifs;
  ifs.open(vector_path, std::ios::binary);
  ifs.read((char *)&M, sizeof(uint32_t));

  cout << "Calc Recall.." << endl
       << "M: " << M << endl
       << "K: " << K << endl
       << "D: " << FILE_D << endl
       << "output: " << output_path << endl;

  if (with_answer) {
    cout << "answer: " << answer_path << endl;
  }

  vectors = (float*)malloc(M * D * sizeof(float));
  memset(vectors, 0, M * D * sizeof(float));
  outputs = (uint32_t*)malloc(M * K * sizeof(uint32_t));
  uint32_t* answers;

  if (with_answer) {
    answers = (uint32_t*)malloc(M * K * sizeof(uint32_t));
  }


  cout << "Loading outputs..." << endl;
  load_knng(output_path, outputs);

  if (with_answer) {
    cout << "Loading answers..." << endl;
    load_knng(answer_path, answers);

    cout << "Calc Recall comparing with answer.." << endl;
    calc_recall(outputs, answers);
  }


  cout << "Loading vectors" << endl;
  float* vec_ptr = vectors;
  for (int i=0; i<M; i++) {
    ifs.read(reinterpret_cast<char*>(vec_ptr), FILE_D * sizeof(float));
    vec_ptr += D;
  }
  ifs.close();

  if (!with_answer){
    cout << "Calc Recall approximately..." << endl;
    vector<thread> recall_threads;

    for (int i=0; i<THREAD; i++) {
      recall_threads.emplace_back(recall_appro_task, APPRO_NUM);
    }

    for (int i=0; i<THREAD; i++) {
      recall_threads[i].join();
    }

    cout <<endl<< "Recall: " << 1.0 * total_hit / THREAD / APPRO_NUM / K << endl;
  }



  cout << "Calc output dis..." << endl;
  calc_dis_mm(vectors, outputs);

  if (with_answer) {
    cout << "Calc answers dis..." << endl;
    calc_dis_mm(vectors, answers);
  }

}
