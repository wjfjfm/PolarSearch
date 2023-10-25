#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>

using namespace std;

uint32_t M;
uint32_t K = 100;
uint32_t D = 200;

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

int main(int argc, char* argv[]) {

  string output_path = "output.bin";
  string answer_path = "result.bin";
  string vector_path = "input.bin";

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
       << "D: " << D << endl
       << "output: " << output_path << endl;

  if (with_answer) {
    cout << "answer: " << answer_path << endl;
  }

  float* vectors = (float*)malloc(M * D * sizeof(float));
  uint32_t* outputs = (uint32_t*)malloc(M * K * sizeof(uint32_t));
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
    ifs.read(reinterpret_cast<char*>(vec_ptr), D * sizeof(float));
    vec_ptr += D;
  }
  ifs.close();

  cout << "Calc output dis..." << endl;
  calc_dis(vectors, outputs);

  if (with_answer) {
    cout << "Calc answers dis..." << endl;
    calc_dis(vectors, answers);
  }
}
