#include <string>
#include <cstdint>

struct GnoimiTrainingConfig {

  int32_t feature_dim;
  int32_t centroids_num; 

  std::string training_data_filename;

};