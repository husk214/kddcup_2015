#include <chrono>
#include <ctime>
#include <string>
#include <cstdlib>
#include <cwctype>

// convert timepoint of system clock to calendar time string
inline std::string as_string(const std::chrono::system_clock::time_point &tp) {
  // convert to system time:
  std::time_t t = std::chrono::system_clock::to_time_t(tp);
  std::string ts = ctime(&t); // convert to calendar time
  ts.resize(ts.size() - 1);   // skip trailing newline
  return ts;
}
// convert calendar time to timepoint of system clock
inline std::chrono::system_clock::time_point
make_time_point(const int year, const int mon = 1, const int day = 1,
                const int hour = 0, const int min = 0, const int sec = 0) {
  struct std::tm t;
  t.tm_sec = sec;          // second of minute (0 .. 59 and 60 for leap seconds)
  t.tm_min = min;          // minute of hour (0 .. 59)
  t.tm_hour = hour;        // hourofday(0..23)
  t.tm_mon = mon - 1;      // day of month (0 .. 31)
  t.tm_mday = day;         // month of year (0 .. 11)
  t.tm_year = year - 1900; // year since 1900
  t.tm_isdst = -1;         // determine whether daylight saving time
  std::time_t tt = std::mktime(&t);
  if (tt == -1) {
    throw "no valid system time";
  }
  return std::chrono::system_clock::from_time_t(tt);
}

// convert calendar time to timepoint of system clock
// str_date = year-month-day-hour-min-sec
inline std::chrono::system_clock::time_point
make_time_point(const std::string &str_date) {
  std::vector<int> date_vec;
  date_vec.push_back(1900); // year
  date_vec.push_back(1);    // month
  date_vec.push_back(1);    // day
  date_vec.push_back(23);    // hour
  date_vec.push_back(59);    // minute
  date_vec.push_back(59);    // second
  int first_index = 0;
  int unit_flag = 0;
  for (size_t i = 0; i <= str_date.size(); ++i) {
    if (!std::iswdigit(str_date[i]) || i == str_date.size()) {
      int tmp = atoi((str_date.substr(first_index, static_cast<int>(i) -
                                                       first_index)).c_str());
      date_vec[unit_flag++] = tmp;
      if (unit_flag > 5)
        break;
      first_index = static_cast<int>(i) + 1;
    }
  }
  struct std::tm t;
  t.tm_sec = date_vec[5];  // second of minute (0 .. 59 and 60 for leap seconds)
  t.tm_min = date_vec[4];  // minute of hour (0 .. 59)
  t.tm_hour = date_vec[3]; // hourofday(0..23)
  t.tm_mon = date_vec[1] - 1;     // day of month (0 .. 31)
  t.tm_mday = date_vec[2];        // month of year (0 .. 11)
  t.tm_year = date_vec[0] - 1900; // year since 1900
  t.tm_isdst = -1;                // determine whether daylight saving time
  std::time_t tt = std::mktime(&t);
  if (tt == -1) {
    throw "no valid system time";
  }
  return std::chrono::system_clock::from_time_t(tt);
}

inline std::time_t make_time_t(const int year, const int mon = 1,
                               const int day = 1, const int hour = 0,
                               const int min = 0, const int sec = 0) {
  struct std::tm t;
  t.tm_sec = sec;          // second of minute (0 .. 59 and 60 for leap seconds)
  t.tm_min = min;          // minute of hour (0 .. 59)
  t.tm_hour = hour;        // hourofday(0..23)
  t.tm_mon = mon - 1;      // day of month (0 .. 31)
  t.tm_mday = day;         // month of year (0 .. 11)
  t.tm_year = year - 1900; // year since 1900
  t.tm_isdst = -1;         // determine whether daylight saving time
  std::time_t tt = std::mktime(&t);
  if (tt == -1) {
    throw "no valid system time";
  }
  return tt;
}
