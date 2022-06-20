#include "common.h"
#include "training_config.h"

DEFINE_int32(feature_dim, 128, "feature dim.");
DEFINE_int32(centroids_num, 256, "centroids num.");
DEFINE_int32(total_training_points, 100000, "total data points num for training.");
DEFINE_int32(trunk_size, 1000, "total data points num in one chunk.");
DEFINE_int32(parallel_degree, 12, "write this code using one Intel-8700K, so you know.");

int D = FLAGS_feature_dim;           // feature dim.
int K = FLAGS_centroids_num;         // K * K centroids in model.
int L = 8; // 
int total_data_points_count = FLAGS_total_training_points; // total data point in training data file.
int total_iteration_count = 10;      // how many in one training process.
int thread_chunk_size = FLAGS_trunk_size;
// total threads to training model.
int threads_count = FLAGS_parallel_degree; // Here, we use Intel 8700K, so set default to 12.
int thread_points_count = total_data_points_count / threads_count;
int thread_chunks_count = thread_points_count / thread_chunk_size; // one thread deal with how many chunks.

std::string training_data_filename = "./sift_learn.fvecs";
std::string init_coarse_filename = "./coarse.fvecs";
std::string init_fine_filename = "./fine.fvecs";
std::string model_file_prefix = "./model/";

int *coarse_assigns = (int *)malloc(total_data_points_count * sizeof(int));
int *fine_assigns = (int *)malloc(total_data_points_count * sizeof(int));

float *alpha = (float *)malloc(K * K * sizeof(float));    // alpha matrix.
float *alpha_num = (float *)malloc(K * K * sizeof(float)); // numerator of alpha.
float *alpha_den = (float *)malloc(K * K * sizeof(float)); // denominator of alpha.
float *fine_vocab = (float *)malloc(D * K * sizeof(float));
float *fine_vocab_num = (float *)malloc(D * K * sizeof(float));
float *fine_vocab_den = (float *)malloc(K * sizeof(float));
float *coarse_vocab = (float *)malloc(D * K * sizeof(float));
float *coarse_vocab_num = (float *)malloc(D * K * sizeof(float));
float *coarse_vocab_den = (float *)malloc(K * sizeof(float));
std::vector<float *> alpha_numerators(threads_count);
std::vector<float *> alpha_denominators(threads_count);
std::vector<float *> fine_vocab_numerators(threads_count);
std::vector<float *> fine_vocab_denominators(threads_count);
std::vector<float *> coarse_vocab_numerators(threads_count);
std::vector<float *> coarse_vocab_denominators(threads_count);

float *coarse_norms = (float *)malloc(K * sizeof(float));
float *fine_norms = (float *)malloc(K * sizeof(float));
float *coarse_fine_product = (float *)malloc(K * K * sizeof(float));
float *errors = (float *)malloc(threads_count * sizeof(float));

void ComputeOptimalAssignsSubset(const int curr_thread_id) {
  int64_t start_id = (total_data_points_count / threads_count) * curr_thread_id;
  float *points_coarse_terms = (float *)malloc(thread_chunk_size * K * sizeof(float));
  float *points_fine_terms = (float *)malloc(thread_chunk_size * K * sizeof(float));
  float *chunk_data_points = (float *)malloc(thread_chunk_size * D * sizeof(float));
  
  errors[curr_thread_id] = 0.0;
  FILE *training_data = fopen(training_data_filename.c_str(), "r");
  fseek(training_data, start_id * (D + 1) * sizeof(float), SEEK_SET);
  
  std::vector<std::pair<float, int>> coarseScores(K);
  for (auto chunk_id = 0; chunk_id < thread_chunks_count; ++chunk_id) {
    std::cout << "[Assings Training] [Thread " << curr_thread_id << "] "
              << "Processing Chunk " << chunk_id << " Of " << thread_chunks_count
              << ", " << chunk_data_points << ", " << thread_chunk_size
              << std::endl;
    fvecs_fread(training_data, chunk_data_points, thread_chunk_size, D);
    fmat_mul_full(coarse_vocab, chunk_data_points, K, thread_chunk_size, D, "TN", points_coarse_terms);
    fmat_mul_full(fine_vocab, chunk_data_points, K, thread_chunk_size, D, "TN", points_fine_terms);
    for (auto point_id = 0; point_id < thread_chunk_size; ++point_id) {
      cblas_saxpy(K, -1.0, coarse_norms, 1, points_coarse_terms + point_id * K, 1);
      for (int k = 0; k < K; ++k) {
        coarseScores[k].first = (-1.0) * points_coarse_terms[point_id * K + k];
        coarseScores[k].second = k;
      }
      std::sort(coarseScores.begin(), coarseScores.end());
      float currentMinScore = 999999999.0;
      int currentMinCoarseId = -1;
      int currentMinFineId = -1;
      for (int l = 0; l < L; ++l) {
        // examine cluster l
        int currentCoarseId = coarseScores[l].second;
        float currentCoarseTerm = coarseScores[l].first;
        for (int currentFineId = 0; currentFineId < K; ++currentFineId)
        {
          float alphaFactor = alpha[currentCoarseId * K + currentFineId];
          float score = currentCoarseTerm + alphaFactor * coarse_fine_product[currentCoarseId * K + currentFineId] +
                        (-1.0) * alphaFactor * points_fine_terms[point_id * K + currentFineId] +
                        alphaFactor * alphaFactor * fine_norms[currentFineId];
          if (score < currentMinScore) {
            currentMinScore = score;
            currentMinCoarseId = currentCoarseId;
            currentMinFineId = currentFineId;
          }
        }
      }
      coarse_assigns[start_id + chunk_id * thread_chunk_size + point_id] = currentMinCoarseId;
      fine_assigns[start_id + chunk_id * thread_chunk_size + point_id] = currentMinFineId;
      errors[curr_thread_id] += currentMinScore * 2 + 1.0; // point has a norm equals 1.0
    }
  }
  fclose(training_data);
  free(chunk_data_points);
  free(points_coarse_terms);
  free(points_fine_terms);
}

void ComputeOptimalAlphaSubset(const int curr_thread_id) {
  memset(alpha_numerators[curr_thread_id], 0, K * K * sizeof(float));
  memset(alpha_denominators[curr_thread_id], 0, K * K * sizeof(float));
  int64_t start_id = (total_data_points_count / threads_count) * curr_thread_id;
  FILE *training_data = fopen(training_data_filename.c_str(), "r");
  fseek(training_data, start_id * (D + 1) * sizeof(float), SEEK_SET);
  float *residual = (float *)malloc(D * sizeof(float));
  float *chunk_data_points = (float *)malloc(thread_chunk_size * D * sizeof(float));
  for (auto chunk_id = 0; chunk_id < thread_chunks_count; ++chunk_id) {
    std::cout << "[Alpha Training] [Thread " << curr_thread_id << "] "
              << "Processing Chunk " << chunk_id << " Of " << thread_chunks_count << std::endl;
    fvecs_fread(training_data, chunk_data_points, thread_chunk_size, D);
    for (auto point_id = 0; point_id < thread_chunk_size; ++point_id) {
      int coarseAssign = coarse_assigns[start_id + chunk_id * thread_chunk_size + point_id];
      int fineAssign = fine_assigns[start_id + chunk_id * thread_chunk_size + point_id];
      memcpy(residual, chunk_data_points + point_id * D, D * sizeof(float));
      cblas_saxpy(D, -1.0, coarse_vocab + coarseAssign * D, 1, residual, 1);
      alpha_numerators[curr_thread_id][coarseAssign * K + fineAssign] +=
          cblas_sdot(D, residual, 1, fine_vocab + fineAssign * D, 1);
      alpha_denominators[curr_thread_id][coarseAssign * K + fineAssign] += fine_norms[fineAssign] * 2; // we keep halves of norms
    }
  }
  fclose(training_data);
  free(chunk_data_points);
  free(residual);
}

void ComputeOptimalFineVocabSubset(const int curr_thread_id) {
  memset(fine_vocab_numerators[curr_thread_id], 0, K * D * sizeof(float));
  memset(fine_vocab_denominators[curr_thread_id], 0, K * sizeof(float));
  int64_t start_id = (total_data_points_count / threads_count) * curr_thread_id;
  FILE *training_data = fopen(training_data_filename.c_str(), "r");
  fseek(training_data, start_id * (D + 1) * sizeof(float), SEEK_SET);
  float *residual = (float *)malloc(D * sizeof(float));
  float *chunk_data_points = (float *)malloc(thread_chunk_size * D * sizeof(float));
  for (auto chunk_id = 0; chunk_id < thread_chunks_count; ++chunk_id) {
    std::cout << "[Fine Vocabs Training] [Thread " << curr_thread_id << "] "
              << "Processing Chunk " << chunk_id << " Of " << thread_chunks_count << std::endl;
    fvecs_fread(training_data, chunk_data_points, thread_chunk_size, D);
    for (auto point_id = 0; point_id < thread_chunk_size; ++point_id)
    {
      int coarseAssign = coarse_assigns[start_id + chunk_id * thread_chunk_size + point_id];
      int fineAssign = fine_assigns[start_id + chunk_id * thread_chunk_size + point_id];
      float alphaFactor = alpha[coarseAssign * K + fineAssign];
      memcpy(residual, chunk_data_points + point_id * D, D * sizeof(float));
      cblas_saxpy(D, -1.0, coarse_vocab + coarseAssign * D, 1, residual, 1);
      cblas_saxpy(D, alphaFactor, residual, 1, fine_vocab_numerators[curr_thread_id] + fineAssign * D, 1);
      fine_vocab_denominators[curr_thread_id][fineAssign] += alphaFactor * alphaFactor;
    }
  }
  fclose(training_data);
  free(chunk_data_points);
  free(residual);
}

void ComputeOptimalCoarseVocabSubset(const int curr_thread_id) {
  memset(coarse_vocab_numerators[curr_thread_id], 0, K * D * sizeof(float));
  memset(coarse_vocab_denominators[curr_thread_id], 0, K * sizeof(float));
  int64_t start_id = (total_data_points_count / threads_count) * curr_thread_id;
  FILE *training_data = fopen(training_data_filename.c_str(), "r");
  fseek(training_data, start_id * (D + 1) * sizeof(float), SEEK_SET);
  float *residual = (float *)malloc(D * sizeof(float));
  float *chunk_data_points = (float *)malloc(thread_chunk_size * D * sizeof(float));
  for (auto chunk_id = 0; chunk_id < thread_chunks_count; ++chunk_id) {
    std::cout << "[Coarse Vocabs Training] [Thread " << curr_thread_id << "] "
              << "Processing Chunk " << chunk_id << " Of " << thread_chunks_count << std::endl;
    fvecs_fread(training_data, chunk_data_points, thread_chunk_size, D);
    for (auto point_id = 0; point_id < thread_chunk_size; ++point_id) {
      int coarseAssign = coarse_assigns[start_id + chunk_id * thread_chunk_size + point_id];
      int fineAssign = fine_assigns[start_id + chunk_id * thread_chunk_size + point_id];
      float alphaFactor = alpha[coarseAssign * K + fineAssign];
      memcpy(residual, chunk_data_points + point_id * D, D * sizeof(float));
      cblas_saxpy(D, -1.0 * alphaFactor, fine_vocab + fineAssign * D, 1, residual, 1);
      cblas_saxpy(D, 1, residual, 1, coarse_vocab_numerators[curr_thread_id] + coarseAssign * D, 1);
      coarse_vocab_denominators[curr_thread_id][coarseAssign] += 1.0;
    }
  }
  fclose(training_data);
  free(chunk_data_points);
  free(residual);
}


void Initialize() {

  // init alpha, fine_vocab, coarse_vocab.
  for (int curr_thread_id = 0; curr_thread_id < threads_count; ++curr_thread_id) {
    alpha_numerators[curr_thread_id] = (float *)malloc(K * K * sizeof(float *));
    alpha_denominators[curr_thread_id] = (float *)malloc(K * K * sizeof(float *));
  }
  for (int curr_thread_id = 0; curr_thread_id < threads_count; ++curr_thread_id) {
    fine_vocab_numerators[curr_thread_id] = (float *)malloc(K * D * sizeof(float *));
    fine_vocab_denominators[curr_thread_id] = (float *)malloc(K * sizeof(float *));
  }
  for (int curr_thread_id = 0; curr_thread_id < threads_count; ++curr_thread_id) {
    coarse_vocab_numerators[curr_thread_id] = (float *)malloc(K * D * sizeof(float));
    coarse_vocab_denominators[curr_thread_id] = (float *)malloc(K * sizeof(float));
  }

  // init vocabs
  fvecs_read(init_coarse_filename.c_str(), D, K, coarse_vocab);
  fvecs_read(init_fine_filename.c_str(), D, K, fine_vocab);
  
  // init alpha
  for (auto i = 0; i < K * K; ++i) {
    alpha[i] = 1.0;
  }
}

void Train() {
  // learn iterations
  std::cout << "Start training iterations..." << std::endl;
  for (auto it = 0; it < total_iteration_count; ++it) {
    for (auto k = 0; k < K; ++k) {
      coarse_norms[k] = cblas_sdot(D, coarse_vocab + k * D, 1, coarse_vocab + k * D, 1) / 2;
      fine_norms[k] = cblas_sdot(D, fine_vocab + k * D, 1, fine_vocab + k * D, 1) / 2;
    }
    fmat_mul_full(fine_vocab, coarse_vocab, K, K, D, "TN", coarse_fine_product);
    // update Assigns
    std::vector<std::thread> workers;
    memset(errors, 0, threads_count * sizeof(float));
    THREADS_POOL_ASSIGN(threads_count, workers, ComputeOptimalAssignsSubset);
    THREADS_POOL_JOIN(threads_count, workers);
    float total_error = 0.0;
    for (int curr_thread_id = 0; curr_thread_id < threads_count; ++curr_thread_id) {
      total_error += errors[curr_thread_id];
    }
    std::cout << "Current reconstruction error... " << total_error / total_data_points_count << std::endl;
    // update alpha
    THREADS_POOL_ASSIGN(threads_count, workers, ComputeOptimalAlphaSubset);
    THREADS_POOL_JOIN(threads_count, workers);
    memset(alpha_num, 0, K * K * sizeof(float));
    memset(alpha_den, 0, K * K * sizeof(float));
    for (int curr_thread_id = 0; curr_thread_id < threads_count; ++curr_thread_id) {
      cblas_saxpy(K * K, 1, alpha_numerators[curr_thread_id], 1, alpha_num, 1);
      cblas_saxpy(K * K, 1, alpha_denominators[curr_thread_id], 1, alpha_den, 1);
    }
    for (int i = 0; i < K * K; ++i) {
      alpha[i] = (alpha_den[i] == 0) ? 1.0 : alpha_num[i] / alpha_den[i];
    }
    // update fine Vocabs
    THREADS_POOL_ASSIGN(threads_count, workers, ComputeOptimalFineVocabSubset);
    THREADS_POOL_JOIN(threads_count, workers);
    memset(fine_vocab_num, 0, K * D * sizeof(float));
    memset(fine_vocab_den, 0, K * sizeof(float));
    for (int curr_thread_id = 0; curr_thread_id < threads_count; ++curr_thread_id) {
      cblas_saxpy(K * D, 1, fine_vocab_numerators[curr_thread_id], 1, fine_vocab_num, 1);
      cblas_saxpy(K, 1, fine_vocab_denominators[curr_thread_id], 1, fine_vocab_den, 1);
    }
    for (int i = 0; i < K * D; ++i) {
      fine_vocab[i] = (fine_vocab_den[i / D] == 0) ? 0 : fine_vocab_num[i] / fine_vocab_den[i / D];
    }
    // update coarse Vocabs
    THREADS_POOL_ASSIGN(threads_count, workers, ComputeOptimalCoarseVocabSubset);
    THREADS_POOL_JOIN(threads_count, workers);
    memset(coarse_vocab_num, 0, K * D * sizeof(float));
    memset(coarse_vocab_den, 0, K * sizeof(float));
    for (int curr_thread_id = 0; curr_thread_id < threads_count; ++curr_thread_id) {
      cblas_saxpy(K * D, 1, coarse_vocab_numerators[curr_thread_id], 1, coarse_vocab_num, 1);
      cblas_saxpy(K, 1, coarse_vocab_denominators[curr_thread_id], 1, coarse_vocab_den, 1);
    }
    for (int i = 0; i < K * D; ++i) {
      coarse_vocab[i] = (coarse_vocab_den[i / D] == 0) ? 0 : coarse_vocab_num[i] / coarse_vocab_den[i / D];
    }
    // save current alpha and vocabs
    std::stringstream alpha_model_filename;
    alpha_model_filename << model_file_prefix << "alpha_" << it << ".dat";
    std::ofstream outAlpha(alpha_model_filename.str().c_str(), std::ios::binary | std::ios::out);
    outAlpha.write((char *)alpha, K * K * sizeof(float));
    outAlpha.close();
    std::stringstream fine_vocab_model_filename;
    fine_vocab_model_filename << model_file_prefix << "fine_" << it << ".dat";
    std::ofstream outFine(fine_vocab_model_filename.str().c_str(),std:: ios::binary | std::ios::out);
    outFine.write((char *)fine_vocab, K * D * sizeof(float));
    outFine.close();
    std::stringstream coarse_vocab_model_filename;
    coarse_vocab_model_filename << model_file_prefix << "coarse_" << it << ".dat";
    std::ofstream outCoarse(coarse_vocab_model_filename.str().c_str(), std::ios::binary | std::ios::out);
    outCoarse.write((char *)coarse_vocab, K * D * sizeof(float));
    outCoarse.close();
  }
  free(coarse_assigns);
  free(fine_assigns);
  free(alpha_num);
  free(alpha_den);
  free(alpha);
  free(coarse_vocab);
  free(coarse_vocab_num);
  free(coarse_vocab_den);
  free(fine_vocab);
  free(fine_vocab_num);
  free(fine_vocab_den);
  free(coarse_norms);
  free(fine_norms);
  free(coarse_fine_product);
  free(errors);
}

int main()
{
  // init alpha, fine_vocab, coarse_vocab
  Initialize(); 
  Train();
  return 0;
}
