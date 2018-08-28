#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

#include <string>
#include <fstream>

std::size_t count_commas_and_newlines(const std::string& filename) {
  std::ifstream file(filename);
  char c;
  std::size_t count = 0;
  while(file.get(c)) {
    if (c == ',' || c == '\n') {
      ++count;
    }
  }
  return count;
}


std::tuple<xt::pyarray<unsigned int>, xt::pyarray<unsigned short>, xt::pyarray<unsigned long>> load_confuslist_index(const std::string &filename) {
  std::size_t array_size = count_commas_and_newlines(filename);

  auto confusion_array = xt::pyarray<unsigned int>::from_shape({array_size});
  auto confusion_word_index_array = xt::pyarray<unsigned short>::from_shape({array_size});
  auto word_anahash_array = xt::pyarray<unsigned long>::from_shape({array_size});

  std::size_t ix = 0;
  unsigned int confusion;
  unsigned short confusion_word_index = 0;
  unsigned long word_anahash;
  char delimiter;

  std::ifstream file(filename);
  while(file >> confusion >> delimiter) {
    if (delimiter == '#') {
      // streaming operator will skip new-lines by default, use noskipws
      while (file >> std::noskipws >> word_anahash >> delimiter) {
        confusion_array[ix] = confusion;
        confusion_word_index_array[ix] = confusion_word_index++;
        word_anahash_array[ix] = word_anahash;
        ++ix;
        if (delimiter == '\n') {
          confusion_word_index = 0;
          break;
        }
      }
    } else {
      throw std::runtime_error("non-# delimiter found after confusion!");
    }
  }
  file.close();

  return {confusion_array, confusion_word_index_array, word_anahash_array};
}

// Python Module and Docstrings

PYBIND11_MODULE(ticcl_output_reader, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        An efficient reader of TICCL output files

        .. currentmodule:: ticcl_output_reader

        .. autosummary::
           :toctree: _generate

           load_confuslist_index
    )pbdoc";

    m.def("load_confuslist_index", load_confuslist_index, "Load a confusion list index file. Returns three NumPy arrays with dtypes uint32, uint16 and uint64 respectively, containing: the confusion value, a running index of the word anahashes corresponding to a common confusion value and the word anahash.");
}
