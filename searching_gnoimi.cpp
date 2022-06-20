#include "common.h"

const int D = 96;
const int K = 16384;
const int M = 8;
const int rerankK = 256;
const int N = 1000000000;
const int num_queries = 10000;
const int L = 32;
const int num_neighbors = 5000;
const int threads_count = 12;

std::string coarse_codebook_filename = "./sift_coarse.fvecs";
std::string fine_codebook_filename = "./sift_fine.fvecs";
std::string alpha_filename = "./sift_alpha.fvecs";
std::string index_filename = "./deep1B_index.dat";
std::string rerank_codes_filename = "./codes_deep1B_8.dat";
std::string rerank_rotation_filename = "./deep1B_rerank_vocabs_rotation8.fvecs";
std::string rerank_vocabs_filename = "./deep1B_rerank_vocabs_8.fvecs";
std::string cell_edges_filename = "./deep1B_cell_edges.dat";
std::string query_filename = "./sift_query.fvecs";
std::string ground_filename = "./sift_groundtruth.ivecs";

struct Record {
  int point_id;
  unsigned char bytes[M];
};

struct Searcher {
  float *coarse_vocab;
  float *coarse_norms;
  float *fine_vocab;
  float *fine_norms;
  float *alpha;
  float *coarse_fine_products;
  Record *index;
  int *cell_edges;
  float *coarse_residuals;
  float *rerank_rotation;
  float *rerank_vocabs;
};

void LoadCellEdgesPart(const std::string &cell_edges_filename,
                       const int start_id, int count, int *cell_edges) {
  std::ifstream input_cell_edges(cell_edges_filename.c_str(), std::ios::binary | std::ios::in);
  input_cell_edges.seekg(start_id * sizeof(int));
  for (int i = 0; i < count; ++i) {
    input_cell_edges.read((char *)&(cell_edges[start_id + i]), sizeof(int));
  }
  input_cell_edges.close();
}

void LoadCellEdges(const std::string &cell_edges_filename, const int N, int *cell_edges) {
  int perThreadCount = N / threads_count;
  std::vector<std::thread> threads_pool;
  for (auto curr_thread_id = 0; curr_thread_id < threads_count; ++curr_thread_id) {
    int start_id = curr_thread_id * perThreadCount;
    int count = (start_id + perThreadCount > N) ? (N - start_id) : perThreadCount;
    threads_pool.push_back(std::thread(std::bind(LoadCellEdgesPart, cell_edges_filename, start_id, count, cell_edges)));
  }
  THREADS_POOL_JOIN(threads_count, threads_pool);
}

void LoadIndexPart(const std::string &index_filename, const std::string &rerank_filename,
                   int startId, int count, Record *index) {
  std::ifstream inputIndex(index_filename.c_str(), std::ios::binary | std::ios::in);
  inputIndex.seekg(startId * sizeof(int));
  std::ifstream inputRerank(rerank_filename.c_str(), std::ios::binary | std::ios::in);
  inputRerank.seekg(startId * sizeof(unsigned char) * M);
  for (int i = 0; i < count; ++i) {
    inputIndex.read((char *)&(index[startId + i].point_id), sizeof(int));
    for (int m = 0; m < M; ++m) {
      inputRerank.read((char *)&(index[startId + i].bytes[m]), sizeof(unsigned char));
    }
  }
  inputIndex.close();
  inputRerank.close();
}

void LoadIndex(const std::string &index_filename, const std::string &rerank_filename, int N, Record *index) {
  int perThreadCount = N / threads_count;
  std::vector<std::thread> threads_pool;
  for (auto thread_id = 0; thread_id < threads_count; ++thread_id) {
    int startId = thread_id * perThreadCount;
    int count = (startId + perThreadCount > N) ? (N - startId) : perThreadCount;
    threads_pool.push_back(std::thread(std::bind(LoadIndexPart, index_filename, rerank_filename, startId, count, index)));
  }
  THREADS_POOL_JOIN(threads_count, threads_pool)
}

void ReadAndPrecomputeVocabsData(Searcher &searcher) {
  searcher.coarse_vocab = (float *)malloc(K * D * sizeof(float));
  fvecs_read(coarse_codebook_filename.c_str(), D, K, searcher.coarse_vocab);
  searcher.fine_vocab = (float *)malloc(K * D * sizeof(float));
  fvecs_read(fine_codebook_filename.c_str(), D, K, searcher.fine_vocab);
  searcher.alpha = (float *)malloc(K * K * sizeof(float));
  fvecs_read(alpha_filename.c_str(), K, K, searcher.alpha);
  searcher.rerank_rotation = (float *)malloc(D * D * sizeof(float));
  fvecs_read(rerank_rotation_filename.c_str(), D, D, searcher.rerank_rotation);
  float *temp = (float *)malloc(K * D * sizeof(float));
  fmat_mul_full(searcher.rerank_rotation, searcher.coarse_vocab,
                D, K, D, "TN", temp);
  memcpy(searcher.coarse_vocab, temp, K * D * sizeof(float));
  free(temp);
  temp = (float *)malloc(K * D * sizeof(float));
  fmat_mul_full(searcher.rerank_rotation, searcher.fine_vocab,
                D, K, D, "TN", temp);
  memcpy(searcher.fine_vocab, temp, K * D * sizeof(float));
  free(temp);
  searcher.coarse_norms = (float *)malloc(K * sizeof(float));
  searcher.fine_norms = (float *)malloc(K * sizeof(float));
  for (int i = 0; i < K; ++i) {
    searcher.coarse_norms[i] = fvec_norm2sqr(searcher.coarse_vocab + D * i, D) / 2;
    searcher.fine_norms[i] = fvec_norm2sqr(searcher.fine_vocab + D * i, D) / 2;
  }
  temp = (float *)malloc(K * K * sizeof(float));
  fmat_mul_full(searcher.coarse_vocab, searcher.fine_vocab,
                K, K, D, "TN", temp);
  searcher.coarse_fine_products = fmat_new_transp(temp, K, K);
  free(temp);
  int Dread;
  std::cout << "Before allocation..." << std::endl;
  searcher.index = (Record *)malloc(N * sizeof(Record));
  LoadIndex(index_filename, rerank_codes_filename, N, searcher.index);
  searcher.cell_edges = (int *)malloc(K * K * sizeof(int));
  LoadCellEdges(cell_edges_filename, N, searcher.cell_edges);
  searcher.coarse_residuals = (float *)malloc(L * D * sizeof(float));
  std::cout << "Before local vocabs reading..." << std::endl;
  searcher.rerank_vocabs = (float *)malloc(rerankK * D * K * sizeof(float));
  fvecs_read(rerank_vocabs_filename.c_str(), D / M, K * M * rerankK, searcher.rerank_vocabs);
  std::cout << "Search data is prepared..." << std::endl;
}

void SearchNearestNeighbors(const Searcher &searcher,
                            const float *queries,
                            int num_neighbors,
                            std::vector<std::vector<std::pair<float, int>>> &result) {
  result.resize(num_queries);
  std::vector<float> queryCoarseDistance(K);
  std::vector<float> query_fineDistance(K);
  std::vector<std::pair<float, int>> coarseList(K);
  for (int qid = 0; qid < num_queries; ++qid) {
    result[qid].resize(num_neighbors, std::make_pair(std::numeric_limits<float>::max(), 0));
  }
  std::vector<int> topPointers(L);
  float *residual = (float *)malloc(D * sizeof(float));
  float *preallocated_vocabs = (float *)malloc(L * rerankK * D * sizeof(float));
  int subDim = D / M;
  std::vector<std::pair<float, int>> scores(L * K);
  std::vector<int> coarseIdToTopId(K);
  std::clock_t c_start = std::clock();

  for (int qid = 0; qid < num_queries; ++qid) {
    // std::cout << qid << std::endl;
    int found = 0;
    fmat_mul_full(searcher.coarse_vocab, queries + qid * D,
                  K, 1, D, "TN", &(queryCoarseDistance[0]));
    fmat_mul_full(searcher.fine_vocab, queries + qid * D,
                  K, 1, D, "TN", &(query_fineDistance[0]));
    for (int c = 0; c < K; ++c) {
      coarseList[c].first = searcher.coarse_norms[c] - queryCoarseDistance[c];
      coarseList[c].second = c;
    }
    std::sort(coarseList.begin(), coarseList.end());
    for (int l = 0; l < L; ++l) {
      int coarseId = coarseList[l].second;
      coarseIdToTopId[coarseId] = l;
      for (int k = 0; k < K; ++k)
      {
        int cellId = coarseId * K + k;
        float alphaFactor = searcher.alpha[cellId];
        scores[l * K + k].first = coarseList[l].first + searcher.fine_norms[k] * alphaFactor * alphaFactor - query_fineDistance[k] * alphaFactor + searcher.coarse_fine_products[cellId] * alphaFactor;
        scores[l * K + k].second = cellId;
      }
      memcpy(searcher.coarse_residuals + l * D, searcher.coarse_vocab + D * coarseId, D * sizeof(float));
      fvec_rev_sub(searcher.coarse_residuals + l * D, queries + qid * D, D);
      memcpy(preallocated_vocabs + l * rerankK * D, searcher.rerank_vocabs + coarseId * rerankK * D, rerankK * D * sizeof(float));
    }
    int cellsCount = num_neighbors * ((float)K * K / N);
    std::nth_element(scores.begin(), scores.begin() + cellsCount, scores.end());
    std::sort(scores.begin(), scores.begin() + cellsCount);
    int currentPointer = 0;
    int cellTraversed = 0;
    while (found < num_neighbors) {
      cellTraversed += 1;
      int cellId = scores[currentPointer].second;
      int topListId = coarseIdToTopId[cellId / K];
      ++currentPointer;
      int cellStart = (cellId == 0) ? 0 : searcher.cell_edges[cellId - 1];
      int cellFinish = searcher.cell_edges[cellId];
      if (cellStart == cellFinish) {
        continue;
      }
      memcpy(residual, searcher.coarse_residuals + topListId * D, D * sizeof(float));
      cblas_saxpy(D, -1.0 * searcher.alpha[cellId], searcher.fine_vocab + (cellId % K) * D, 1, residual, 1);
      float *cell_vocab = preallocated_vocabs + topListId * rerankK * D;
      for (int id = cellStart; id < cellFinish && found < num_neighbors; ++id) {
        result[qid][found].second = searcher.index[id].point_id;
        result[qid][found].first = 0.0;
        float diff = 0.0;
        for (int m = 0; m < M; ++m) {
          float *codeword = cell_vocab + m * rerankK * subDim + searcher.index[id].bytes[m] * subDim;
          float *residualSubvector = residual + m * subDim;
          for (int d = 0; d < subDim; ++d)
          {
            diff = residualSubvector[d] - codeword[d];
            result[qid][found].first += diff * diff;
          }
        }
        ++found;
      }
    }
    std::sort(result[qid].begin(), result[qid].end());
  }
  std::clock_t c_end = std::clock();
  std::cout << std::fixed << std::setprecision(2) << "CPU time used: "
            << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC / num_queries << " ms\n";
  free(residual);
  free(searcher.coarse_vocab);
  free(searcher.coarse_norms);
  free(searcher.fine_vocab);
  free(searcher.fine_norms);
  free(searcher.alpha);
  free(searcher.index);
  free(searcher.cell_edges);
  free(searcher.coarse_fine_products);
}

float ComputeRecallAt(const std::vector<std::vector<std::pair<float, int>>> &result,
                      const int *groundtruth, const int R) {
  int limit = (R < result[0].size()) ? R : result[0].size();
  int positive = 0;
  for (int i = 0; i < result.size(); ++i) {
    for (int j = 0; j < limit; ++j) {
      if (result[i][j].second == groundtruth[i]) {
        ++positive;
      }
    }
  }
  return (float(positive) / result.size());
}

void ComputeRecall(const std::vector<std::vector<std::pair<float, int>>> &result,
                   const int *groundtruth) {
  for (auto i = 0; i < 11; ++i) {
    int R = std::pow(2.0, i);
    std::cout << std::fixed << std::setprecision(5) << "Recall@ " << R << ": " << ComputeRecallAt(result, groundtruth, R) << "\n";
  }
}

int main() {
  Searcher searcher;
  ReadAndPrecomputeVocabsData(searcher);
  float *queries = (float *)malloc(num_queries * D * sizeof(float));
  fvecs_read(query_filename.c_str(), D, num_queries, queries);
  float *temp = (float *)malloc(num_queries * D * sizeof(float));
  fmat_mul_full(searcher.rerank_rotation, queries,
                D, num_queries, D, "TN", temp);
  memcpy(queries, temp, num_queries * D * sizeof(float));
  free(temp);
  std::vector<std::vector<std::pair<float, int>>> result;
  SearchNearestNeighbors(searcher, queries, num_neighbors, result);
  std::cout << "Before reading groundtruth..." << std::endl;
  int *groundtruth = (int *)malloc(num_queries * 1 * sizeof(int));
  int d;
  ivecs_new_read(ground_filename.c_str(), &d, &groundtruth);
  std::cout << "Groundtruth is read..." << std::endl;
  ComputeRecall(result, groundtruth);
  return 0;
}
