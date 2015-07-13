/******************************************
* Compile
g++ -I/usr/include/eigen3 make_train_test_data.cpp -o make_train_test_data

g++ -std=c++11 -I../eigen3.2.1 make_train_test_data.cpp -o make_train_test_data

* run
./make_train_test_data dataset/train/log_train.csv dataset/train/truth_train.csv dataset/train/enrollment_train.csv dataset/test/log_test.csv dataset/test/enrollment_test.csv dataset/date.csv dataset/train/3.csv dataset/test/3.csv 3 1

******************************************/
#include <utility>
#include <string>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <iomanip>
#include <unordered_map>
#include <vector>

// #include <boost/algorithm/string.hpp>
// #include <boost/foreach.hpp>

#include <Eigen/Core>
#include "timepoint.hpp"

using mymap = std::unordered_map<std::string, int>;
using mypair = std::pair<std::string, int>;
using mymap2 = std::unordered_map<int, std::string>;
using mypair2 = std::pair<int, std::string>;
using pair_sort = std::pair<mymap::iterator, int>;

constexpr int the_number_of_features_ = 1000;
constexpr int course_number = 39;
constexpr int max_moving_average = 30;
constexpr int max_course_interval = 30;

constexpr double one_over_24 = (1.0 / 24.0);
constexpr double one_over_60 = (1.0 / 60.0);

static std::string s_problem = "problem";
static std::string s_video = "video";
static std::string s_access = "access";
static std::string s_wiki = "wiki";
static std::string s_discussion = "discussion";
static std::string s_navigate = "navigate";
static std::string s_page_close = "page_close";
static std::string s_server = "server";
static std::string s_browser = "browser";

std::array<int, max_moving_average> total_mm, problem_mm, video_mm, access_mm,
    wiki_mm, discussion_mm, navigate_mm, page_close_mm;

std::array<int, max_course_interval> total_ma, problem_ma, video_ma, access_ma,
    wiki_ma, discussion_ma, navigate_ma, page_close_ma;

inline bool lessPair(const pair_sort &l, const pair_sort &r) {
  return l.second < r.second;
}
inline bool greaterPair(const pair_sort &l, const pair_sort &r) {
  return l.second > r.second;
}

inline std::vector<std::string> split_string(const std::string &str,
                                             const std::string &delim) {
  std::vector<std::string> res;
  std::string::size_type current = 0, found, delimlen = delim.size();
  while ((found = str.find(delim, current)) != std::string::npos) {
    res.push_back(std::string(str, current, found - current));
    current = found + delimlen;
  }
  res.push_back(std::string(str, current, str.size() - current));
  return res;
}

template <typename RealType>
inline double mean(const std::vector<RealType> &vec) {
  double sum = 0.0;
  if (!vec.empty()) {
    for (auto i = vec.begin(); i != vec.end(); ++i)
      sum += *i;
    sum /= static_cast<RealType>(vec.size());
  }
  return sum;
}

template <typename RealType>
inline double vari(const std::vector<RealType> &vec, const double &the_mean) {
  double variance = 0.0;
  if (!vec.empty()) {
    for (auto i = vec.begin(); i != vec.end(); ++i)
      variance += (*i - the_mean) * (*i - the_mean);
    variance /= static_cast<RealType>(vec.size());
  }
  return variance;
}

template <typename RealType>
inline double median(std::vector<RealType> &vec, int &longest_interval) {
  std::sort(vec.begin(), vec.end());
  int l = vec.size();
  if (l > 2) {
    longest_interval = vec[vec.size() - 1];
    double tmp = (l - 1) * 0.5;
    int lowerIndex = int(floor(tmp));
    int upperIndex = int(ceil(tmp));
    tmp -= lowerIndex;
    return (1.0 - tmp) * vec[lowerIndex] + tmp * vec[upperIndex];
  } else {
    longest_interval = 0;
    return 0.0;
  }
}

template <typename durationType>
inline bool check_dif_chrono(const std::chrono::system_clock::time_point &sctp1,
                             const std::chrono::system_clock::time_point &sctp2,
                             const int &dif) {
  if (abs(static_cast<int>(std::chrono::duration_cast<durationType>(
                               sctp2 - sctp1).count())) >= dif) {
    return true;
  }
  return false;
}

template <typename durationType>
inline int get_dif_chrono(const std::chrono::system_clock::time_point &sctp1,
                          const std::chrono::system_clock::time_point &sctp2) {
  return abs(static_cast<int>(
      std::chrono::duration_cast<durationType>(sctp2 - sctp1).count()));
}

int naive_atoi(const char *p) {
  int r = 0;
  bool neg = false;
  while (*p == ' ')
    ++p;
  if (*p == '-') {
    neg = true;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  while (*p >= '0' && *p <= '9') {
    r = (r * 10.0) + (*p - '0');
    ++p;
  }
  if (neg) {
    r = -r;
  }
  return r;
}

template <typename ValueType> ValueType naive_atot(const char *p) {
  ValueType r = 0.0;
  bool neg = false;
  while (*p == ' ')
    ++p;
  if (*p == '-') {
    neg = true;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  while (*p >= '0' && *p <= '9') {
    r = (r * 10.0) + (*p - '0');
    ++p;
  }
  if (*p == '.') {
    ValueType f = 0.0;
    int n = 0;
    ++p;
    while (*p >= '0' && *p <= '9') {
      f = (f * 10.0) + (*p - '0');
      ++p;
      ++n;
    }
    r += f / std::pow(10.0, n);
  }
  if (neg) {
    r = -r;
  }
  return r;
}

template <typename ValueType> ValueType naive_atot(const std::string &s) {
  ValueType r = 0.0;
  bool neg = false;
  auto it = s.begin();
  while (*it == ' ')
    ++it;
  if (*it == '-') {
    neg = true;
    ++it;
  } else if (*it == '-') {
    ++it;
  }
  while (*it >= '0' && *it <= '9') {
    r = (r * 10.0) + (*it - '0');
    ++it;
  }
  if (*it == '.') {
    ++it;
    ValueType f = 0.0;
    int n = 0;
    while (*it >= '0' && *it <= '9') {
      f = (f * 10.0) + (*it - '0');
      ++it;
      ++n;
    }
    r += f / std::pow(10.0, n);
  }
  if (neg)
    r = -r;
  return r;
}

template <typename ValueType, int _Cols>
bool save(const Eigen::Matrix<ValueType, Eigen::Dynamic, _Cols> &mat,
          const std::string &file_name, const std::string &header = "none") {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "cannot open the file for writing in save\n";
    return false;
  }
  if (header != "none") {
    output_file << header << std::endl;
  }
  if (_Cols == 1) {
    for (int i = 0; i < mat.rows(); ++i)
      output_file << mat.coeffRef(i, 0) << std::endl;

  } else {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        output_file << mat.coeffRef(i, j) << " ";
      }
      output_file << std::endl;
    }
  }
  return true;
}

template <typename ValueType, int _Cols>
bool save_csv(const Eigen::Matrix<ValueType, Eigen::Dynamic, _Cols> &mat,
              const std::string &file_name, const std::string &header = "none",
              const int max_mat_cols = -1) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "cannot open the file for writing in save\n";
    return false;
  }
  if (header != "none") {
    output_file << header << std::endl;
  }
  if (_Cols == 1) {
    for (int i = 0; i < mat.rows(); ++i)
      output_file << mat.coeffRef(i, 0) << std::endl;

  } else {
    int matcols = 0;
    if (max_mat_cols == -1) {
      matcols = mat.cols() - 1;
    } else {
      matcols = max_mat_cols - 1;
    }
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < matcols; ++j) {
        output_file << mat.coeffRef(i, j) << ",";
      }
      output_file << mat.coeffRef(i, matcols) << std::endl;
    }
  }
  return true;
}

template <typename ValueType, int _Cols>
bool save_libsvm(const Eigen::Matrix<ValueType, Eigen::Dynamic, _Cols> &mat,
                 const std::string &file_name,
                 const std::string &header = "none",
                 const int max_mat_cols = -1) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "cannot open the file for writing in save\n";
    return false;
  }
  if (header != "none") {
    output_file << header << std::endl;
  }
  if (_Cols == 1) {
    for (int i = 0; i < mat.rows(); ++i)
      output_file << mat.coeffRef(i, 0) << std::endl;

  } else {
    int matcols = 0;
    if (max_mat_cols == -1) {
      matcols = mat.cols() - 1;
    } else {
      matcols = max_mat_cols - 1;
    }
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < matcols; ++j) {
        output_file << mat.coeffRef(i, j) << ",";
      }
      output_file << mat.coeffRef(i, matcols) << std::endl;
    }
  }
  return true;
}

void update_count_time(const int &h, std::array<int, 8> &count_arr) {
  if (h < 3) {
    ++count_arr[0];
  } else if (h < 6) {
    ++count_arr[1];
  } else if (h < 9) {
    ++count_arr[2];
  } else if (h < 12) {
    ++count_arr[3];
  } else if (h < 15) {
    ++count_arr[4];
  } else if (h < 18) {
    ++count_arr[5];
  } else if (h < 21) {
    ++count_arr[6];
  } else {
    ++count_arr[7];
  }
}

void update_event(const std::string &event_name,
                  std::array<int, 7> &event_arr) {
  if (event_name == s_access) {
    ++event_arr[2];
  } else if (event_name == s_video) {
    ++event_arr[1];
  } else if (event_name == s_wiki) {
    ++event_arr[3];
  } else if (event_name == s_discussion) {
    ++event_arr[4];
  } else if (event_name == s_navigate) {
    ++event_arr[5];
  } else if (event_name == s_page_close) {
    ++event_arr[6];
  } else {
    ++event_arr[0];
  }
}

inline std::chrono::system_clock::time_point get_time_point(int &for_hour) {
  int y = naive_atoi(strtok(nullptr, "-"));
  int mo = naive_atoi(strtok(nullptr, "-"));
  int d = naive_atoi(strtok(nullptr, "T"));
  for_hour = naive_atoi(strtok(nullptr, ":"));
  int mi = naive_atoi(strtok(nullptr, ":"));
  int s = naive_atoi(strtok(nullptr, ","));
  // std::cout << "gtp : " << y << " " << mo << " " << d << " " << for_hour << "
  // "
  //           << mi << " " << s << std::endl;
  return make_time_point(y, mo, d, for_hour, mi, s);
}

std::vector<bool> get_not_remove_flag(std::string rstr, int feature_num) {
  std::vector<bool> flag_vec(feature_num + 1, true);
  std::vector<std::string> split_vec;
  split_vec = split_string(rstr, " ");
  if (!split_vec.empty()) {
    if (split_vec[0] != "") {
      for (auto &&el : split_vec) {
        int rn = naive_atoi(el.c_str());
        if (rn >= 1 && rn <= feature_num) {
          flag_vec[rn - 1] = false;
        } else {
          std::cerr << "error : not exist feature " << rn << std::endl;
        }
      }
    }
  }
  return flag_vec;
}

void clean_event_mm_ma(void) {
  total_mm.fill(0);
  problem_mm.fill(0);
  video_mm.fill(0);
  access_mm.fill(0);
  wiki_mm.fill(0);
  discussion_mm.fill(0);
  navigate_mm.fill(0);
  page_close_mm.fill(0);
  total_ma.fill(0);
  problem_ma.fill(0);
  video_ma.fill(0);
  access_ma.fill(0);
  wiki_ma.fill(0);
  discussion_ma.fill(0);
  navigate_ma.fill(0);
  page_close_ma.fill(0);
}

void update_event_mm_ma(const std::string &event_name, const int diff_day,
                        const int ma_index = -1) {
  int index_day = std::max(max_moving_average - 1 - diff_day, 0);
  if (ma_index == -1) {
    ++total_mm[index_day];
    if (event_name == s_access) {
      ++access_mm[index_day];
    } else if (event_name == s_video) {
      ++video_mm[index_day];
    } else if (event_name == s_wiki) {
      ++wiki_mm[index_day];
    } else if (event_name == s_discussion) {
      ++discussion_mm[index_day];
    } else if (event_name == s_navigate) {
      ++navigate_mm[index_day];
    } else if (event_name == s_page_close) {
      ++page_close_mm[index_day];
    } else {
      ++problem_mm[index_day];
    }
  } else {
    ++total_mm[index_day];
    ++total_ma[ma_index];
    if (event_name == s_access) {
      ++access_mm[index_day];
      ++access_ma[ma_index];
    } else if (event_name == s_video) {
      ++video_mm[index_day];
      ++video_ma[ma_index];
    } else if (event_name == s_wiki) {
      ++wiki_mm[index_day];
      ++wiki_ma[ma_index];
    } else if (event_name == s_discussion) {
      ++discussion_mm[index_day];
      ++discussion_ma[ma_index];
    } else if (event_name == s_navigate) {
      ++navigate_mm[index_day];
      ++navigate_ma[ma_index];
    } else if (event_name == s_page_close) {
      ++page_close_mm[index_day];
      ++page_close_ma[ma_index];
    } else {
      ++problem_mm[index_day];
      ++problem_ma[ma_index];
    }
  }
}

std::vector<int> get_course_bit(const int id_number,
                                const int max_bit_num = 39) {
  std::vector<int> vec(max_bit_num);
  for (int i = 0; i < max_bit_num; ++i)
    vec[i] = (i == id_number);
  return vec;
}

std::vector<double>
get_moving_average(const std::array<int, max_moving_average> &event_mm,
                   const int interval, const int shift) {
  std::vector<double> vec;
  double sum = 0.0;
  int first_index = 0;
  for (; first_index + interval < max_moving_average; first_index += shift) {
    sum = 0.0;
    for (int j = 0; j <= interval; ++j)
      sum += event_mm[first_index + j];
    vec.push_back(sum / interval);
  }
  sum = 0.0;
  for (int i = first_index; i < max_moving_average; ++i)
    sum += event_mm[i];
  vec.push_back(sum / (max_moving_average - first_index - 1));
  return vec;
}

std::vector<std::vector<double>> get_all_moving_average(const int interval,
                                                        const int shift) {
  std::vector<std::vector<double>> vv;
  vv.push_back(get_moving_average(total_ma, interval, shift));
  vv.push_back(get_moving_average(problem_ma, interval, shift));
  vv.push_back(get_moving_average(video_ma, interval, shift));
  vv.push_back(get_moving_average(access_ma, interval, shift));
  vv.push_back(get_moving_average(wiki_ma, interval, shift));
  vv.push_back(get_moving_average(discussion_ma, interval, shift));
  vv.push_back(get_moving_average(navigate_ma, interval, shift));
  vv.push_back(get_moving_average(page_close_ma, interval, shift));
  return vv;
}

char *sdm_line = nullptr;
int sdm_max_line_len;
char *readline(FILE *input) {
  if (fgets(sdm_line, sdm_max_line_len, input) == nullptr)
    return nullptr;

  while (strrchr(sdm_line, '\n') == nullptr) {
    sdm_max_line_len *= 2;
    sdm_line = (char *)realloc(sdm_line, sdm_max_line_len);
    int len = (int)strlen(sdm_line);
    if (fgets(sdm_line + len, sdm_max_line_len - len, input) == nullptr)
      break;
  }

  return sdm_line;
}

std::vector<double>
get_weighted_sum(const std::array<int, max_course_interval> &event_array) {
  std::vector<double> v;
  double tmp_sum = 0.0, tmp_sum_1 = 0.0, tmp_sum_2 = 0.0, last10_sum = 0.0,
         last10_sum_1 = 0.0, last10_sum_2 = 0.0;
  double non_zero_day = 0.0;
  for (unsigned long i = 0; i < max_course_interval; ++i) {
    tmp_sum += (i + 1) * event_array[i];
    if (i > 20 && event_array[i] > 0) {
      last10_sum += (i - 19) * event_array[i];
      last10_sum_1 +=
          (i - 19) * (double(event_array[i]) / (event_array[i] + 1.0));
      last10_sum_2 += (i - 19) * (i - 19) *
                      (double(event_array[i]) / (event_array[i] + 1.0));
    }
    tmp_sum_1 +=
        double(i + 1) * (double(event_array[i]) / (event_array[i] + 1.0));
    tmp_sum_2 += double(i + 1) * (i + 1) *
                 (double(event_array[i]) / (event_array[i] + 1.0));
    if (event_array[i] > 0)
      ++non_zero_day;
  }
  v.push_back(tmp_sum);
  v.push_back(tmp_sum_1);
  v.push_back(tmp_sum_2);
  v.push_back(last10_sum);
  v.push_back(last10_sum_1);
  v.push_back(last10_sum_2);
  v.push_back(non_zero_day);
  return v;
}

std::vector<std::vector<double>> get_all_event_weighted_sum() {
  std::vector<std::vector<double>> vv;
  vv.push_back(get_weighted_sum(total_ma));
  vv.push_back(get_weighted_sum(problem_ma));
  vv.push_back(get_weighted_sum(video_ma));
  vv.push_back(get_weighted_sum(access_ma));
  vv.push_back(get_weighted_sum(wiki_ma));
  vv.push_back(get_weighted_sum(discussion_ma));
  vv.push_back(get_weighted_sum(navigate_ma));
  vv.push_back(get_weighted_sum(page_close_ma));
  return vv;
}

std::vector<double> get_primal_interaction(const std::vector<double> &vec) {
  std::vector<double> pi;
  for (int i = 0; i < vec.size(); ++i)
    for (int j = i+1; j < vec.size(); ++j)
      pi.push_back(vec[i]*vec[j]);
    return pi;
}

struct course_begin_end_date {
  std::chrono::system_clock::time_point begin_date;
  std::chrono::system_clock::time_point end_date;
};

int main(int argc, char const *argv[]) {
  using namespace std;

  if (argc != 11) {
    cout << "Usage : > ./make_train_valid log_train.csv true_trian.csv "
            "enrollment_train.csv"
            "log_test.csv enrollment_test.csv data.csv output_file_name_train "
            "output_file_name_test" << endl;
  } else {
    std::string log_train_file_name = argv[1];
    std::string label_name = argv[2];
    std::string enrollment_train_name = argv[3];
    std::string log_test_file_name = argv[4];
    std::string enrollment_file_name = argv[5];
    std::string date_csv_file_name = argv[6];
    std::string output_file_name = argv[7];
    std::string output_file_name_test = argv[8];
    std::string str_ma_interval = argv[9];
    std::string str_ma_shift = argv[10];
    int ma_interval = naive_atot<int>(str_ma_interval);
    int ma_shift = naive_atot<int>(str_ma_shift);

    FILE *fp = fopen(log_train_file_name.c_str(), "r");
    if (fp == nullptr) {
      std::cerr << "file open error : " << log_train_file_name << std::endl;
      return -1;
    }

    FILE *fp_test = fopen(log_test_file_name.c_str(), "r");
    if (fp_test == nullptr) {
      std::cerr << "file open error : " << log_test_file_name << std::endl;
      return -1;
    }

    ifstream ifs(label_name);
    if (!ifs.is_open()) {
      std::cerr << "file open error : " << label_name << std::endl;
      return -1;
    }

    ifstream ifs_train_eid(enrollment_train_name);
    if (!ifs_train_eid.is_open()) {
      std::cerr << "file open error : " << enrollment_train_name << std::endl;
      return -1;
    }

    ifstream ifs_test_eid(enrollment_file_name);
    if (!ifs_test_eid.is_open()) {
      std::cerr << "file open error : " << enrollment_file_name << std::endl;
      return -1;
    }
    std::string cltr = "./dataset/complete_lecture_train.csv";
    ifstream ifs_train_comp_lec(cltr);
    if (!ifs_train_comp_lec.is_open()) {
      std::cerr << "file open error : "
                << cltr << std::endl;
      return -1;
    }

    std::string clte = "./dataset/complete_lecture_test.csv";
    ifstream ifs_test_comp_lec(clte);
    if (!ifs_test_comp_lec.is_open()) {
      std::cerr << "file open error : "
                << clte << std::endl;
      return -1;
    }

    std::string detr = "./dataset/numbers_depth.csv";
    ifstream ifs_train_depth(detr);
    if (!ifs_train_depth.is_open()) {
      std::cerr << "file open error : "
                << detr << std::endl;
      return -1;
    }

    std::string dete = "./dataset/numbers_depth_test.csv";
    ifstream ifs_test_depth(dete);
    if (!ifs_test_depth.is_open()) {
      std::cerr << "file open error : "
                << dete << std::endl;
      return -1;
    }

    ifstream ifs_date_csv(date_csv_file_name);
    if (!ifs_date_csv.is_open()) {
      std::cerr << "file open error : " << date_csv_file_name << std::endl;
      return -1;
    }

    std::ofstream output_file(output_file_name);
    if (!output_file.is_open()) {
      std::cerr << "cannot open the file for writing in save : "
                << output_file_name << std::endl;
      return -1;
    }

    std::ofstream output_file_test(output_file_name_test);
    if (!output_file_test.is_open()) {
      std::cerr << "cannot open the file for writing in save : "
                << output_file_name_test << std::endl;
      return -1;
    }
    cout << "end open file" << endl;
    // const char *deliminator = ",\n";
    char *p;
    std::string enrollment_id, tmp_eid;
    std::string the_word, event, event_source;

    array<int, 7> event_array;
    event_array.fill(0);

    int tmp_hour = 0;
    array<int, 8> count_time;
    count_time.fill(0);

    std::chrono::system_clock::time_point tp, pre_tp, min3_tp, min30_tp,
        hour3_tp, day1_tp, pre_video_tp, base_tp;
    base_tp = make_time_point(2014, 8, 1, 0, 0, 0);
    int total_event = 0;
    int min3_event = 0;
    int min30_event = 0;
    int hour3_event = 0;
    int day1_event = 0;
    int server_source_event = 0;
    int browser_source_event = 0;

    std::vector<double> interval_day;
    double interval_day_mean = 0.0;
    bool first_video_flag = false;
    std::vector<double> video_interval_hour;
    std::vector<int> output_course_bit_vec;

    sdm_max_line_len = 1024;
    sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

    int diff_base_tp = 0;
    int eid_count, eid_num;
    bool first_flag = true;

    std::chrono::system_clock::time_point the_eid_last_tp;
    std::vector<std::vector<double>> movinge_average_vv;
    int max_feature = 0;
    mymap uid_course_num_map;
    mymap2 eid_uid_map;

    std::chrono::system_clock::time_point tmp_tp, first_tp, last_tp,
        the_course_end_tp;

    std::vector<double> primal_interaction_vec;
    /**************************************
    * process for complete lecture and depth
    ***************************************/
    std::vector<std::vector<double>> comp_lec_train;
    std::vector<std::vector<double>> depth_train;
    std::vector<std::vector<double>> comp_lec_test;
    std::vector<std::vector<double>> depth_test;
    std::string tmp_cd_str;
    std::vector<std::string> tmp_cd_str_vec;
    std::vector<double> tmp_cd_vec;
    getline(ifs_train_comp_lec, tmp_cd_str); // skip header
    while (getline(ifs_train_comp_lec, tmp_cd_str)) {
      tmp_cd_str_vec = split_string(tmp_cd_str, ",");
      tmp_cd_vec.clear();
      for (int tmpi = 1; tmpi < tmp_cd_str_vec.size(); ++tmpi) {
        if (tmpi != 1)
          tmp_cd_vec.push_back(naive_atot<double>(tmp_cd_str_vec[tmpi]));
      }
      comp_lec_train.push_back(tmp_cd_vec);
    }
    getline(ifs_train_depth, tmp_cd_str); // skip header
    while (getline(ifs_train_depth, tmp_cd_str)) {
      tmp_cd_str_vec = split_string(tmp_cd_str, ",");
      tmp_cd_vec.clear();
      for (int tmpi = 1; tmpi < tmp_cd_str_vec.size(); ++tmpi) {
        if (tmpi != 2)
          tmp_cd_vec.push_back(naive_atot<double>(tmp_cd_str_vec[tmpi]));
      }
      depth_train.push_back(tmp_cd_vec);
    }
    getline(ifs_test_comp_lec, tmp_cd_str); // skip header
    while (getline(ifs_test_comp_lec, tmp_cd_str)) {
      tmp_cd_str_vec = split_string(tmp_cd_str, ",");
      tmp_cd_vec.clear();
      for (int tmpi = 1; tmpi < tmp_cd_str_vec.size(); ++tmpi) {
        if (tmpi != 1)
          tmp_cd_vec.push_back(naive_atot<double>(tmp_cd_str_vec[tmpi]));
      }
      comp_lec_test.push_back(tmp_cd_vec);
    }
    getline(ifs_test_depth, tmp_cd_str); // skip header
    while (getline(ifs_test_depth, tmp_cd_str)) {
      tmp_cd_str_vec = split_string(tmp_cd_str, ",");
      tmp_cd_vec.clear();
      for (int tmpi = 1; tmpi < tmp_cd_str_vec.size(); ++tmpi) {
        if (tmpi != 2)
          tmp_cd_vec.push_back(naive_atot<double>(tmp_cd_str_vec[tmpi]));
      }
      depth_test.push_back(tmp_cd_vec);
    }

    /**************************************
    * check course begin and end date
    ***************************************/
    std::string str_course_date;
    getline(ifs_date_csv, str_course_date); // skip header
    std::vector<std::string> course_data_vec;
    std::unordered_map<std::string, course_begin_end_date> map_course_date;
    while (getline(ifs_date_csv, str_course_date)) {
      course_data_vec = split_string(str_course_date, ",");
      first_tp = make_time_point(course_data_vec[1]);
      last_tp = make_time_point(course_data_vec[2]);
      course_begin_end_date tmp_cbed = {first_tp, last_tp};
      map_course_date.insert(std::pair<string, course_begin_end_date>(
          course_data_vec[0], tmp_cbed));
    }
    /**************************************
    * check first and last timepoint
    ***************************************/
    std::vector<double> train_first_last_interval;
    std::vector<std::chrono::system_clock::time_point> train_last_tp_vec;
    readline(fp); // skip header

    // first process
    readline(fp);
    enrollment_id = strtok(sdm_line, ",");

    while (1) {
      if (readline(fp) != nullptr) {
        tmp_eid = strtok(sdm_line, ",");
      } else {
        tmp_eid = "end";
      }
      if (tmp_eid == "\n") {
        continue;
      }
      if (tmp_eid != enrollment_id) {
        train_first_last_interval.push_back(
            get_dif_chrono<std::chrono::hours>(last_tp, first_tp) *
            one_over_24);
        train_last_tp_vec.push_back(last_tp);
        if (tmp_eid == "end") {
          break;
        }
        first_flag = true;
        first_tp = get_time_point(tmp_hour);
        last_tp = first_tp;
        enrollment_id = tmp_eid;
      }
      if (first_flag) {
        first_flag = false;
      } else {
        last_tp = get_time_point(tmp_hour);
      }
    }

    fclose(fp);
    fp = fopen(log_train_file_name.c_str(), "r");

    // test

    std::vector<double> test_first_last_interval;
    std::vector<std::chrono::system_clock::time_point> test_last_tp_vec;
    readline(fp_test); // skip header

    // first process
    readline(fp_test);
    enrollment_id = strtok(sdm_line, ",");

    while (1) {
      if (readline(fp_test) != nullptr) {
        tmp_eid = strtok(sdm_line, ",");
      } else {
        tmp_eid = "end";
      }
      if (tmp_eid == "\n") {
        continue;
      }
      if (tmp_eid != enrollment_id) {
        test_first_last_interval.push_back(
            get_dif_chrono<std::chrono::hours>(last_tp, first_tp) *
            one_over_24);
        test_last_tp_vec.push_back(last_tp);
        if (tmp_eid == "end") {
          break;
        }
        first_flag = true;
        first_tp = get_time_point(tmp_hour);
        last_tp = first_tp;
        enrollment_id = tmp_eid;
      }
      if (first_flag) {
        first_flag = false;
      } else {
        last_tp = get_time_point(tmp_hour);
      }
    }

    fclose(fp_test);
    fp_test = fopen(log_test_file_name.c_str(), "r");
    /************************************
    * course id process
    *************************************/
    mymap course_map;
    mymap2 eid_course_map;
    std::string train_course, train_eid_str, trian_uid_str;
    int course_id_number = 0;

    getline(ifs_train_eid, train_course); // skip header
    while (getline(ifs_train_eid, train_eid_str, ',')) {
      getline(ifs_train_eid, trian_uid_str, ',');
      auto find_it_uc = uid_course_num_map.find(trian_uid_str);
      if (find_it_uc == uid_course_num_map.end()) {
        uid_course_num_map[trian_uid_str] = 1;
      } else {
        find_it_uc->second = find_it_uc->second + 1;
      }
      getline(ifs_train_eid, train_course);
      train_course.erase(--train_course.end());
      auto find_it = course_map.find(train_course);

      if (find_it == course_map.end()) {
        course_map[train_course] = course_id_number;
        ++course_id_number;
      }
      int train_eid = naive_atot<int>(train_eid_str);
      eid_course_map[train_eid] = train_course;
      eid_uid_map[train_eid] = trian_uid_str;
    }

    /************************************
    * enrollment id test process
    *************************************/
    std::vector<int> eid_vec;
    string eid_tmp_str, eid_tmp_str1, test_uid_str;
    mymap2 test_eid_course_map;

    getline(ifs_test_eid, eid_tmp_str); // skip header
    while (getline(ifs_test_eid, eid_tmp_str, ',')) {
      getline(ifs_test_eid, test_uid_str, ',');
      auto find_it_uc = uid_course_num_map.find(test_uid_str);
      if (find_it_uc == uid_course_num_map.end()) {
        uid_course_num_map[test_uid_str] = 1;
      } else {
        find_it_uc->second = find_it_uc->second + 1;
      }
      getline(ifs_test_eid, eid_tmp_str1);
      eid_tmp_str1.erase(--eid_tmp_str1.end());
      std::remove(eid_tmp_str1.begin(), eid_tmp_str1.end(), ' ');
      int test_eid = naive_atot<int>(eid_tmp_str);
      eid_vec.push_back(test_eid);
      test_eid_course_map[test_eid] = eid_tmp_str1;
      eid_uid_map[test_eid] = test_uid_str;
    }

    /************************************
    * train data process
    *************************************/

    readline(fp); // skip header

    std::vector<double> label_vec;
    string label_tmp_str;
    while (getline(ifs, label_tmp_str, ',')) {
      getline(ifs, label_tmp_str);
      label_vec.push_back(naive_atot<double>(label_tmp_str));
    }
    eid_count = 0;
    eid_num = label_vec.size();
    Eigen::MatrixXd dataset_mat(eid_num, the_number_of_features_ + 1);
    std::chrono::system_clock::time_point the_eid_first_tp;

    // first process
    readline(fp);
    enrollment_id = strtok(sdm_line, ",");
    auto tmp_find_it = eid_course_map.find(naive_atot<int>(enrollment_id));
    auto tmp_course_fe_date = map_course_date[tmp_find_it->second];
    the_course_end_tp = tmp_course_fe_date.end_date;

    pre_tp = get_time_point(tmp_hour);
    tp = pre_tp;
    the_eid_first_tp = tp;
    update_count_time(tmp_hour, count_time);

    ++total_event;
    min3_tp = pre_tp;
    ++min3_event;
    min30_tp = pre_tp;
    ++min30_event;
    hour3_tp = pre_tp;
    ++hour3_event;
    day1_tp = pre_tp;
    ++day1_event;

    event_source = strtok(nullptr, ",");
    if (event_source == s_server) {
      ++server_source_event;
    } else {
      ++browser_source_event;
    }

    event = strtok(nullptr, ",");
    update_event(event, event_array);

    the_eid_last_tp = train_last_tp_vec[0];
    update_event_mm_ma(
        event, static_cast<int>(
                   get_dif_chrono<std::chrono::hours>(pre_tp, the_eid_last_tp) *
                   one_over_24));
    if (event == s_video) {
      pre_video_tp = pre_tp;
      first_video_flag = true;
    }

    while (1) {
      if (readline(fp) != nullptr) {
        tmp_eid = strtok(sdm_line, ",");
      } else {
        tmp_eid = "end";
      }
      if (tmp_eid == "\n") {
        continue;
      }
      if (tmp_eid != enrollment_id) {
        int feature_index = 0;
        dataset_mat.coeffRef(eid_count, feature_index++) = min3_event;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            max(0.0, log(min3_event));
        dataset_mat.coeffRef(eid_count, feature_index++) = min30_event;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            max(0.0, log(min30_event));
        dataset_mat.coeffRef(eid_count, feature_index++) = hour3_event;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            max(0.0, log(hour3_event));
        dataset_mat.coeffRef(eid_count, feature_index++) = day1_event;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            max(0.0, log(day1_event));

        for (int i = 0; i < count_time.size(); ++i) {
          dataset_mat.coeffRef(eid_count, feature_index++) = count_time[i];
          dataset_mat.coeffRef(eid_count, feature_index++) =
              max(0.0, log(count_time[i]));
        }

        int longest_interval = 0;
        double interval_day_mean = mean(interval_day);
        double interval_day_vari = vari(interval_day, interval_day_mean);
        double interval_day_median = median(interval_day, longest_interval);

        dataset_mat.coeffRef(eid_count, feature_index++) = interval_day_mean;
        dataset_mat.coeffRef(eid_count, feature_index++) = interval_day_vari;
        dataset_mat.coeffRef(eid_count, feature_index++) = interval_day_median;
        dataset_mat.coeffRef(eid_count, feature_index++) = longest_interval;

        int video_longest_interval = 0;
        double video_interval_mean = mean(video_interval_hour);
        double video_interval_vari =
            vari(video_interval_hour, video_interval_mean);
        double video_interval_median =
            median(video_interval_hour, video_longest_interval);

        dataset_mat.coeffRef(eid_count, feature_index++) = video_interval_mean;
        dataset_mat.coeffRef(eid_count, feature_index++) = video_interval_vari;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            video_interval_median;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            video_longest_interval;

        for (int i = 0; i < event_array.size(); ++i) {
          dataset_mat.coeffRef(eid_count, feature_index++) = event_array[i];
          dataset_mat.coeffRef(eid_count, feature_index++) =
              max(0.0, log(event_array[i]));
        }
        dataset_mat.coeffRef(eid_count, feature_index++) = total_event;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            max(0.0, log(total_event));
        dataset_mat.coeffRef(eid_count, feature_index++) = server_source_event;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            max(0.0, log(server_source_event));
        dataset_mat.coeffRef(eid_count, feature_index++) = browser_source_event;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            max(0.0, log(browser_source_event));

        diff_base_tp = get_dif_chrono<std::chrono::hours>(tp, base_tp);
        dataset_mat.coeffRef(eid_count, feature_index++) =
            diff_base_tp * one_over_24;

        int train_eid = naive_atot<int>(enrollment_id);
        auto find_it_uc = eid_uid_map.find(train_eid);
        auto find_it_uc2 = uid_course_num_map.find(find_it_uc->second);
        double num_courced = find_it_uc2->second;
        dataset_mat.coeffRef(eid_count, feature_index++) = num_courced;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            train_first_last_interval[eid_count];
        // course feature
        auto find_it = eid_course_map.find(train_eid);
        auto find_it2 = course_map.find(find_it->second);
        output_course_bit_vec = get_course_bit(find_it2->second);
        for (int ci = 0; ci < course_number; ++ci)
          dataset_mat.coeffRef(eid_count, feature_index++) =
              output_course_bit_vec[ci];

        // movinge_average_vv = get_all_moving_average(6, 3);
        // for (auto tmpvec : movinge_average_vv)
        //   for (auto tmpele : tmpvec)
        //     dataset_mat.coeffRef(eid_count, feature_index++) = tmpele;

        auto the_course_fe_date = map_course_date[find_it->second];
        double begin_to_log_first =
            double(get_dif_chrono<std::chrono::hours>(
                the_eid_first_tp, the_course_fe_date.begin_date)) *
            one_over_24;
        double log_end_to_end =
            double(get_dif_chrono<std::chrono::hours>(
                the_course_fe_date.end_date, the_eid_last_tp)) *
            one_over_24;

        dataset_mat.coeffRef(eid_count, feature_index++) = begin_to_log_first;
        dataset_mat.coeffRef(eid_count, feature_index++) = log_end_to_end;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            begin_to_log_first - log_end_to_end;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            log_end_to_end - begin_to_log_first;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            begin_to_log_first * log_end_to_end;
        dataset_mat.coeffRef(eid_count, feature_index++) =
            fabs(begin_to_log_first - log_end_to_end) *
            (31.0 - train_first_last_interval[eid_count]);

        std::vector<std::vector<double>> vv_weighted =
            get_all_event_weighted_sum();
        for (auto tmpvec : vv_weighted)
          for (auto tmpele : tmpvec)
            dataset_mat.coeffRef(eid_count, feature_index++) = tmpele;

        movinge_average_vv = get_all_moving_average(ma_interval, ma_shift);
        for (auto tmpvec : movinge_average_vv)
          for (auto tmpele : tmpvec)
            dataset_mat.coeffRef(eid_count, feature_index++) = tmpele;

        for (auto tmpele : comp_lec_train[eid_count])
          dataset_mat.coeffRef(eid_count, feature_index++) = tmpele;
        for (auto tmpele : depth_train[eid_count])
          dataset_mat.coeffRef(eid_count, feature_index++) = tmpele;

        // primal interaction
        primal_interaction_vec.push_back(max(0.0, log(min30_event)));
        primal_interaction_vec.push_back(interval_day_median);
        primal_interaction_vec.push_back(longest_interval);
        primal_interaction_vec.push_back(train_first_last_interval[eid_count]);
        primal_interaction_vec.push_back(begin_to_log_first);
        primal_interaction_vec.push_back(log_end_to_end);
        primal_interaction_vec.push_back((vv_weighted[0])[2]);
        primal_interaction_vec.push_back((vv_weighted[0])[5]);
        primal_interaction_vec.push_back((vv_weighted[0])[6]);
        primal_interaction_vec.push_back((comp_lec_train[eid_count])[2]);

        std::vector<double> pi_vec =
            get_primal_interaction(primal_interaction_vec);

        for (auto tmpele : pi_vec)
          dataset_mat.coeffRef(eid_count, feature_index++) = tmpele;

        // the last column is label (dropout)
        dataset_mat.coeffRef(eid_count, feature_index++) = label_vec[eid_count];

        max_feature = std::max(max_feature, feature_index);
        // initilize
        ++eid_count;
        enrollment_id = tmp_eid;
        event_array.fill(0);
        count_time.fill(0);
        total_event = 0;
        min3_event = 0;
        min30_event = 0;
        hour3_event = 0;
        day1_event = 0;
        interval_day.clear();
        first_video_flag = false;
        video_interval_hour.clear();
        server_source_event = 0;
        browser_source_event = 0;
        clean_event_mm_ma();
        primal_interaction_vec.clear();
        first_flag = true;
        if (tmp_eid == "end") {
          break;
        }
        tmp_find_it = eid_course_map.find(naive_atot<int>(enrollment_id));
        tmp_course_fe_date = map_course_date[tmp_find_it->second];
        the_course_end_tp = tmp_course_fe_date.end_date;
        the_eid_last_tp = train_last_tp_vec[eid_count];
        tp = get_time_point(tmp_hour);
        the_eid_first_tp = tp;
        update_count_time(tmp_hour, count_time);
        min3_tp = tp;
        min30_tp = tp;
        hour3_tp = tp;
        day1_tp = tp;
      }
      ++total_event;
      if (!first_flag) {
        tp = get_time_point(tmp_hour);
        update_count_time(tmp_hour, count_time);
      } else {
        first_flag = false;
      }
      if (check_dif_chrono<std::chrono::minutes>(min3_tp, tp, 3)) {
        min3_tp = tp;
        ++min3_event;
      }
      if (check_dif_chrono<std::chrono::minutes>(min30_tp, tp, 30)) {
        min30_tp = tp;
        ++min30_event;
      }
      if (check_dif_chrono<std::chrono::hours>(hour3_tp, tp, 3)) {
        hour3_tp = tp;
        ++hour3_event;
      }
      int diff_hours = get_dif_chrono<std::chrono::hours>(day1_tp, tp);
      if (diff_hours >= 24) {
        interval_day.push_back(double(diff_hours) * one_over_24);
        day1_tp = tp;
        ++day1_event;
      }
      event_source = strtok(nullptr, ",");
      if (event_source == s_server) {
        ++server_source_event;
      } else {
        ++browser_source_event;
      }
      event = strtok(nullptr, ",");
      update_event(event, event_array);
      update_event_mm_ma(
          event, static_cast<int>(
                     get_dif_chrono<std::chrono::hours>(tp, the_eid_last_tp) *
                     one_over_24),
          static_cast<int>(
              std::max(max_course_interval - 1 -
                           static_cast<int>(get_dif_chrono<std::chrono::hours>(
                                                tp, the_course_end_tp) *
                                            one_over_24),
                       0)));

      if (event == s_video) {
        if (first_video_flag) {
          int diff_video_hours =
              get_dif_chrono<std::chrono::minutes>(pre_video_tp, tp);
          video_interval_hour.push_back(double(diff_video_hours) * one_over_60);
          pre_video_tp = tp;
        } else {
          pre_video_tp = tp;
          first_video_flag = true;
        }
      }
    }
    cout << "ok train" << endl;
    first_flag = true;

    /************************************
    * test data process
    *************************************/
    readline(fp_test); // skip header

    eid_count = 0;
    eid_num = eid_vec.size();
    Eigen::MatrixXd dataset_mat_test(eid_num, the_number_of_features_ + 1);

    // first process
    readline(fp_test);
    enrollment_id = strtok(sdm_line, ",");
    tmp_find_it = test_eid_course_map.find(naive_atot<int>(enrollment_id));
    tmp_course_fe_date = map_course_date[tmp_find_it->second];
    the_course_end_tp = tmp_course_fe_date.end_date;
    pre_tp = get_time_point(tmp_hour);
    update_count_time(tmp_hour, count_time);
    tp = pre_tp;
    the_eid_first_tp = tp;
    ++total_event;
    min3_tp = pre_tp;
    ++min3_event;
    min30_tp = pre_tp;
    ++min30_event;
    hour3_tp = pre_tp;
    ++hour3_event;
    day1_tp = pre_tp;
    ++day1_event;

    event_source = strtok(nullptr, ",");
    if (event_source == s_server) {
      ++server_source_event;
    } else {
      ++browser_source_event;
    }

    event = strtok(nullptr, ",");
    update_event(event, event_array);

    the_eid_last_tp = test_last_tp_vec[0];
    update_event_mm_ma(
        event, static_cast<int>(
                   get_dif_chrono<std::chrono::hours>(pre_tp, the_eid_last_tp) *
                   one_over_24));
    if (event == s_video) {
      pre_video_tp = pre_tp;
      first_video_flag = true;
    }

    while (1) {
      if (readline(fp_test) != nullptr) {
        tmp_eid = strtok(sdm_line, ",");
      } else {
        tmp_eid = "end";
      }
      if (tmp_eid == "\n") {
        continue;
      }
      if (tmp_eid != enrollment_id) {
        int feature_index = 0;
        dataset_mat_test.coeffRef(eid_count, feature_index++) = min3_event;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            max(0.0, log(min3_event));
        dataset_mat_test.coeffRef(eid_count, feature_index++) = min30_event;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            max(0.0, log(min30_event));
        dataset_mat_test.coeffRef(eid_count, feature_index++) = hour3_event;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            max(0.0, log(hour3_event));
        dataset_mat_test.coeffRef(eid_count, feature_index++) = day1_event;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            max(0.0, log(day1_event));

        for (int i = 0; i < count_time.size(); ++i) {
          dataset_mat_test.coeffRef(eid_count, feature_index++) = count_time[i];
          dataset_mat_test.coeffRef(eid_count, feature_index++) =
              max(0.0, log(count_time[i]));
        }

        int longest_interval = 0;
        double interval_day_mean = mean(interval_day);
        double interval_day_vari = vari(interval_day, interval_day_mean);
        double interval_day_median = median(interval_day, longest_interval);

        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            interval_day_mean;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            interval_day_vari;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            interval_day_median;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            longest_interval;

        int video_longest_interval = 0;
        double video_interval_mean = mean(video_interval_hour);
        double video_interval_vari =
            vari(video_interval_hour, video_interval_mean);
        double video_interval_median =
            median(video_interval_hour, video_longest_interval);

        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            video_interval_mean;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            video_interval_vari;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            video_interval_median;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            video_longest_interval;

        for (int i = 0; i < event_array.size(); ++i) {
          dataset_mat_test.coeffRef(eid_count, feature_index++) =
              event_array[i];
          dataset_mat_test.coeffRef(eid_count, feature_index++) =
              max(0.0, log(event_array[i]));
        }
        dataset_mat_test.coeffRef(eid_count, feature_index++) = total_event;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            max(0.0, log(total_event));
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            server_source_event;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            max(0.0, log(server_source_event));
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            browser_source_event;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            max(0.0, log(browser_source_event));
        diff_base_tp = get_dif_chrono<std::chrono::hours>(tp, base_tp);
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            diff_base_tp * one_over_24;

        int test_eid = naive_atot<int>(enrollment_id);
        auto find_it_uc = eid_uid_map.find(test_eid);
        auto find_it_uc2 = uid_course_num_map.find(find_it_uc->second);
        double num_courced = find_it_uc2->second;
        dataset_mat_test.coeffRef(eid_count, feature_index++) = num_courced;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            test_first_last_interval[eid_count];
        // course feature
        auto find_it = test_eid_course_map.find(test_eid);
        auto find_it2 = course_map.find(find_it->second);
        output_course_bit_vec = get_course_bit(find_it2->second);
        for (int ci = 0; ci < course_number; ++ci)
          dataset_mat_test.coeffRef(eid_count, feature_index++) =
              output_course_bit_vec[ci];

        // movinge_average_vv = get_all_moving_average(6, 3);
        // for (auto tmpvec : movinge_average_vv)
        //   for (auto tmpele : tmpvec)
        //     dataset_mat_test.coeffRef(eid_count, feature_index++) = tmpele;

        auto the_course_fe_date = map_course_date[find_it->second];
        double begin_to_log_first =
            double(get_dif_chrono<std::chrono::hours>(
                the_eid_first_tp, the_course_fe_date.begin_date)) *
            one_over_24;
        double log_end_to_end =
            double(get_dif_chrono<std::chrono::hours>(
                the_course_fe_date.end_date, the_eid_last_tp)) *
            one_over_24;

        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            begin_to_log_first;
        dataset_mat_test.coeffRef(eid_count, feature_index++) = log_end_to_end;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            begin_to_log_first - log_end_to_end;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            log_end_to_end - begin_to_log_first;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            begin_to_log_first * log_end_to_end;
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            fabs(begin_to_log_first - log_end_to_end) *
            (31.0 - test_first_last_interval[eid_count]);

        std::vector<std::vector<double>> vv_weighted =
            get_all_event_weighted_sum();
        for (auto tmpvec : vv_weighted)
          for (auto tmpele : tmpvec)
            dataset_mat_test.coeffRef(eid_count, feature_index++) = tmpele;

        movinge_average_vv = get_all_moving_average(ma_interval, ma_shift);
        for (auto tmpvec : movinge_average_vv)
          for (auto tmpele : tmpvec)
            dataset_mat_test.coeffRef(eid_count, feature_index++) = tmpele;

        for (auto tmpele : comp_lec_test[eid_count])
          dataset_mat_test.coeffRef(eid_count, feature_index++) = tmpele;
        for (auto tmpele : depth_test[eid_count])
          dataset_mat_test.coeffRef(eid_count, feature_index++) = tmpele;

        primal_interaction_vec.push_back(max(0.0, log(min30_event)));
        primal_interaction_vec.push_back(interval_day_median);
        primal_interaction_vec.push_back(longest_interval);
        primal_interaction_vec.push_back(test_first_last_interval[eid_count]);
        primal_interaction_vec.push_back(begin_to_log_first);
        primal_interaction_vec.push_back(log_end_to_end);
        primal_interaction_vec.push_back((vv_weighted[0])[2]);
        primal_interaction_vec.push_back((vv_weighted[0])[5]);
        primal_interaction_vec.push_back((vv_weighted[0])[6]);
        primal_interaction_vec.push_back((comp_lec_test[eid_count])[2]);

        std::vector<double> pi_vec =
            get_primal_interaction(primal_interaction_vec);

        for (auto tmpele : pi_vec)
          dataset_mat_test.coeffRef(eid_count, feature_index++) = tmpele;

        // the last column is enrollment id
        dataset_mat_test.coeffRef(eid_count, feature_index++) =
            eid_vec[eid_count];

        // initilize
        ++eid_count;
        enrollment_id = tmp_eid;
        event_array.fill(0);
        count_time.fill(0);
        total_event = 0;
        min3_event = 1;
        min30_event = 1;
        hour3_event = 1;
        day1_event = 1;
        interval_day.clear();
        first_video_flag = false;
        video_interval_hour.clear();
        server_source_event = 0;
        browser_source_event = 0;
        clean_event_mm_ma();
        primal_interaction_vec.clear();
        first_flag = true;
        if (tmp_eid == "end") {
          break;
        }
        tmp_find_it = test_eid_course_map.find(naive_atot<int>(enrollment_id));
        tmp_course_fe_date = map_course_date[tmp_find_it->second];
        the_course_end_tp = tmp_course_fe_date.end_date;
        the_eid_last_tp = test_last_tp_vec[eid_count];
        tp = get_time_point(tmp_hour);
        the_eid_first_tp = tp;
        update_count_time(tmp_hour, count_time);
        min3_tp = tp;
        min30_tp = tp;
        hour3_tp = tp;
        day1_tp = tp;
      }
      ++total_event;
      if (!first_flag) {
        tp = get_time_point(tmp_hour);
        update_count_time(tmp_hour, count_time);
      } else {
        first_flag = false;
      }
      if (check_dif_chrono<std::chrono::minutes>(min3_tp, tp, 3)) {
        min3_tp = tp;
        ++min3_event;
      }
      if (check_dif_chrono<std::chrono::minutes>(min30_tp, tp, 30)) {
        min30_tp = tp;
        ++min30_event;
      }
      if (check_dif_chrono<std::chrono::hours>(hour3_tp, tp, 3)) {
        hour3_tp = tp;
        ++hour3_event;
      }
      int diff_hours = get_dif_chrono<std::chrono::hours>(day1_tp, tp);
      if (diff_hours >= 24) {
        interval_day.push_back(double(diff_hours) * one_over_24);
        day1_tp = tp;
        ++day1_event;
      }
      event_source = strtok(nullptr, ",");
      if (event_source == s_server) {
        ++server_source_event;
      } else {
        ++browser_source_event;
      }
      event = strtok(nullptr, ",");
      update_event(event, event_array);
      update_event_mm_ma(
          event, static_cast<int>(
                     get_dif_chrono<std::chrono::hours>(tp, the_eid_last_tp) *
                     one_over_24),
          static_cast<int>(
              std::max(max_course_interval - 1 -
                           static_cast<int>(get_dif_chrono<std::chrono::hours>(
                                                tp, the_course_end_tp) *
                                            one_over_24),
                       0)));

      if (event == s_video) {
        if (first_video_flag) {
          int diff_video_hours =
              get_dif_chrono<std::chrono::minutes>(pre_video_tp, tp);
          // cout << diff_video_hours << std::endl;
          video_interval_hour.push_back(double(diff_video_hours) * one_over_60);
          pre_video_tp = tp;
        } else {
          pre_video_tp = tp;
          first_video_flag = true;
        }
      }
    }
    save_csv(dataset_mat, output_file_name, "none", max_feature);
    save_csv(dataset_mat_test, output_file_name_test, "none", max_feature);
    fclose(fp);

    free(sdm_line);
  }
  return 0;
}
