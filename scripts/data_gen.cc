#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <queue>
#include <chrono>

struct Vector {
    std::vector<float> elements;
};

// 自定义比较函数，按照pair的double值升序排序
struct IdDisPair {
  
  IdDisPair(uint32_t _id, double _dis) : id(_id), dis(_dis) {}
  bool operator<(const IdDisPair& other) const {
      return dis < other.dis;
  }
  uint32_t id;
  double dis;
};

template<typename T>
class TopKQueue {
public:
  TopKQueue(int length) : maxLength(length) {}

  // 插入元素
  void insert(const T& element) {
    pq.push(element);
    if (pq.size() > maxLength) {
      pq.pop();
    }
  }

  std::deque<T> dumpQueue() {
    std::deque<T> result;
    while (!pq.empty()) {
      result.push_front(pq.top());
      pq.pop();
    }

    return std::move(result);
  }

private:
  std::priority_queue<T> pq;
  int maxLength;
};


double calcDis(const Vector& v1, const Vector& v2) {
  double sum = 0.0f;
  for (int i = 0; i < v1.elements.size(); i++) {
    double diff = v1.elements[i] - v2.elements[i];
    sum += diff * diff;
  }
  return sum;
}

// 生成一个随机向量
Vector genRandVec(int dimensions) {
    Vector v;
    for (int i = 0; i < dimensions; i++) {
        float randomValue = static_cast<float>(rand()) / RAND_MAX;
        v.elements.push_back(randomValue);
    }
    return std::move(v);
}

void writeFloat32VectorToFile(const std::vector<Vector>& vectors, const std::string& filename, const bool with_num) {
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    if (with_num) {
      uint32_t vec_num = vectors.size();
      file.write(reinterpret_cast<const char*>(&vec_num), sizeof(uint32_t));
    }

    for (const auto& vector : vectors) {
        for (const auto& element : vector.elements) {
            file.write(reinterpret_cast<const char*>(&element), sizeof(float));
        }
    }

    file.close();
}

int main(int argc, char* argv[]) {
  int M = 2000; 
  int N = 1000;
  int dimensions = 200;
  int K = 100;

  if (argc > 1) {
    M = std::atoi(argv[1]);
  }

  if (argc > 2) {
    N = std::atoi(argv[2]);
  }

  std::cout << "Random gen with"
	     << " M: " << M 
	     << ", N: " << N
	     << ", K: " << K
	     << ", D: " << dimensions
	     << std::endl;

  Vector targetVector = genRandVec(dimensions); // 随机生成一个目标向量
  std::vector<Vector> randomVectors;
  std::vector<Vector> targetVectors;

  for (int i = 0; i < M; i++) {
    randomVectors.push_back(genRandVec(dimensions));
  }

  for (int i = 0; i < N; i++) {
    targetVectors.push_back(genRandVec(dimensions));
  }

  writeFloat32VectorToFile(randomVectors, "random_vectors.bin", true);
  writeFloat32VectorToFile(targetVectors, "target_vectors.bin", true);

  std::cout << "Start Running..." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  TopKQueue<IdDisPair> topk_que(K);
  for (int n = 0; n < N; n++) {
    auto target_vector = targetVectors[n];
    for (int m = 0; m < M; m++) {
      topk_que.insert(IdDisPair(m, calcDis(randomVectors[m], target_vector)));
    }

    auto knn_queue = topk_que.dumpQueue();
    std::vector<Vector> knn_vectors;
    for (const auto & id_dis : knn_queue) {
      knn_vectors.push_back(randomVectors[id_dis.id]);
    }
    writeFloat32VectorToFile(knn_vectors , "knn_vectors.bin", false);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  int minutes = duration / 60000;
  int seconds = (duration % 60000) / 1000;
  int milliseconds = duration % 1000;

  std::cout << "Time Use:" << minutes << " m " << seconds << " s " << milliseconds << " ms" << std::endl;

  return 0;
}
