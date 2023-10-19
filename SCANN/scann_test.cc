
#include <iostream>
#include <vector>
#include <scann.h>

int main() {
    // 创建一个包含数据的矩阵
    std::vector<std::vector<float>> data = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    // 配置并训练Scann模型
    scann::Scann<float> scann;
    scann.Init(scann::ScannConfig());
    scann.AddDataset(data);

    // 指定KNN搜索的参数
    scann::ScannConfig::NearestNeighborsConfig knn_config;
    knn_config.k = 2; // 搜索最近的2个邻居

    // 执行KNN搜索
    std::vector<int> query = {1.5, 2.5, 3.5}; // 查询向量
    std::vector<scann::ScannResult> results = scann.ComputeNeighbors(query, knn_config);

    // 打印结果
    for (const auto& result : results) {
        std::cout << "Nearest neighbor index: " << result.id << ", distance: " << result.distance << std::endl;
    }

    return 0;
}
