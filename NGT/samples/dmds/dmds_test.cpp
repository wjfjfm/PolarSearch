
#include	"NGT/NGTQ/QuantizedGraph.h"
#include <chrono>

#define K 100
#define D 200

float calcDis(vector<float> & A, vector<float> & B) {
  double sum;
  for (int i=0; i<A.size(); i++) {
    double diff = A[i] - B[i];
    sum += diff * diff;
  }
  return sum;
}

int
main(int argc, char **argv)
{
  string	indexPath	= "index";
  string	objectFile	= "./data/random_vectors.bin";
  string	resultFile	= "./data/knn_graphs.bin";

  // index construction
  NGT::Property	property;
  property.dimension = D;
  property.edgeSizeForCreation = K+100;
  property.batchSizeForCreation = 256;
  property.objectType = NGT::ObjectSpace::ObjectType::Float;
  property.distanceType = NGT::Index::Property::DistanceType::DistanceTypeL2;
  property.graphType = NGT::Property::GraphType::GraphTypeKNNG;
  property.threadPoolSize = 31;

  std::cout << "creating the index framework..." << std::endl;
  NGT::Index::create(indexPath, property);
  NGT::Index	index(indexPath);

  std::cout << "appending the objects..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  ifstream file(objectFile, std::ios::binary);
  uint32_t length;
  file.read(reinterpret_cast<char*>(&length), sizeof(length));
  cout << "lenght: " << length << endl;
  std::vector<float> obj;
  vector<std::vector<float>> all_vec;
  obj.resize(D);
  for (uint32_t i = 0; i<length; i++) {
    file.read(reinterpret_cast<char*>(obj.data()), D * sizeof(float));
    index.insert(obj);
    all_vec.push_back(obj);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  cout << "Time use: " << elapsedTime << endl;


  start = std::chrono::high_resolution_clock::now();

  std::cout << "building the index..." << std::endl;
  index.createIndex(32);

  end = std::chrono::high_resolution_clock::now();
  elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  cout << "Time use: " << elapsedTime << endl;

  // quantization
  // size_t dimensionOfSubvector = 1;
  // size_t maxNumberOfEdges = D;
  // try {
  //   std::cout << "quantizing the index..." << std::endl;
  //   NGTQG::Index::quantize(indexPath, dimensionOfSubvector, maxNumberOfEdges, true);
  // } catch (NGT::Exception &err) {
  //   cerr << "Error " << err.what() << endl;
  //   return 1;
  // } catch (...) {
  //   cerr << "Error" << endl;
  //   return 1;
  // }

  // nearest neighbor search
  std::cout << "searching the index..." << std::endl;

  ifstream rsfile(resultFile, std::ios::binary);

  file.seekg(4);
  std::vector<float> query;
  query.resize(D);
  std::vector<uint32_t> result;
  result.resize(K);
  size_t succ_cnt = 0;


  start = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i<length; i++) {
    file.read(reinterpret_cast<char*>(query.data()), D * sizeof(float));
    NGT::SearchQuery		sc(query);
    NGT::ObjectDistances	objects;
    sc.setResults(&objects);
    sc.setSize(K);
    sc.setEpsilon(0);

    index.search(sc);

    rsfile.read(reinterpret_cast<char*>(result.data()), K * sizeof(uint32_t));
    // for (auto item: result) {
    //   cout << item << ":" << calcDis(all_vec[item], query) << " " ;
    // }
    // cout << endl;

    unordered_set<uint32_t> check(result.begin(), result.end());

    for (size_t i = 0; i < objects.size(); i++) {
      // cout << objects[i].id - 1 << ":" << objects[i].distance << ":" << calcDis(all_vec[objects[i].id - 1], query);
      if (check.count(objects[i].id - 1)) {
        succ_cnt++;
      }
    }

    // cout << endl << endl;

  }

  end = std::chrono::high_resolution_clock::now();
  elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  cout << "Time use: " << elapsedTime << endl;
  double qps = 1000.0 * length / elapsedTime;
  cout << "QPS: " << qps << endl;

  float recall = 100.0 * succ_cnt / (1.0 * length * K);
  cout << endl << "Recall: " << recall << "%" << endl;

  return 0;
}
