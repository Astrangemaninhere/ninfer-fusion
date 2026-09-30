// k40cxx17 prototype (a): a C++17 back-fill of std::span.
//
// Injected with `-include k40_shim/cxx17_compat.h` so that NO tree source has to
// change: the 907 `std::span` uses keep compiling verbatim.
//
// WHY IT IS NEEDED: std::span is C++20. libstdc++-11's <span> is written as
//   #if __cplusplus > 201703L
//     ... everything ...
//   #endif
// so under -std=c++17 `#include <span>` succeeds and defines NOTHING, and every
// use is "namespace std has no member span".
//
// WHY IT IS NOT FREE: this declares a new name inside namespace std, which
// [namespace.std] forbids. The program is ill-formed, no diagnostic required.
// No ODR violation actually arises here because the real <span> is empty in
// this dialect (__cpp_lib_span is undefined), but this is a deliberate,
// documented extension of std, not standard code. Flagged in the report.
#ifndef K40_CXX17_COMPAT_H
#define K40_CXX17_COMPAT_H

#if defined(__cplusplus) && (__cplusplus < 202002L) && !defined(__cpp_lib_span)

#include <cstddef>
#include <array>
#include <iterator>
#include <type_traits>
#include <utility>

namespace std {
inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

template <class T, std::size_t Extent = dynamic_extent>
class span {
 public:
  using element_type = T;
  using value_type = typename std::remove_cv<T>::type;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;

  static constexpr std::size_t extent = Extent;

  constexpr span() noexcept : data_(nullptr), size_(0) {}

  constexpr span(pointer ptr, size_type count) noexcept : data_(ptr), size_(count) {}
  constexpr span(pointer first, pointer last) noexcept
      : data_(first), size_(static_cast<size_type>(last - first)) {}

  template <std::size_t N>
  constexpr span(element_type (&arr)[N]) noexcept : data_(arr), size_(N) {}

  template <class U, std::size_t N,
            class = typename std::enable_if<
                std::is_convertible<U (*)[], T (*)[]>::value>::type>
  constexpr span(std::array<U, N>& arr) noexcept : data_(arr.data()), size_(N) {}

  template <class U, std::size_t N,
            class = typename std::enable_if<
                std::is_convertible<const U (*)[], T (*)[]>::value>::type>
  constexpr span(const std::array<U, N>& arr) noexcept : data_(arr.data()), size_(N) {}

  /* any container with .data() and .size(): vector, string, raw buffers */
  template <class C,
            class = decltype(static_cast<pointer>(std::declval<C&>().data())),
            class = decltype(std::declval<C&>().size()),
            class = typename std::enable_if<
                !std::is_same<typename std::remove_cv<C>::type, span>::value>::type>
  constexpr span(C& c) noexcept
      : data_(c.data()), size_(static_cast<size_type>(c.size())) {}

  template <class U, std::size_t E,
            class = typename std::enable_if<
                std::is_convertible<U (*)[], T (*)[]>::value>::type>
  constexpr span(const span<U, E>& other) noexcept
      : data_(other.data()), size_(other.size()) {}

  constexpr span(const span&) noexcept = default;
  constexpr span& operator=(const span&) noexcept = default;

  constexpr iterator begin() const noexcept { return data_; }
  constexpr iterator end() const noexcept { return data_ + size_; }
  constexpr const_iterator cbegin() const noexcept { return data_; }
  constexpr const_iterator cend() const noexcept { return data_ + size_; }
  constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
  constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }

  constexpr reference front() const { return data_[0]; }
  constexpr reference back() const { return data_[size_ - 1]; }
  constexpr reference operator[](size_type i) const { return data_[i]; }
  constexpr pointer data() const noexcept { return data_; }
  constexpr size_type size() const noexcept { return size_; }
  constexpr size_type size_bytes() const noexcept { return size_ * sizeof(element_type); }
  constexpr bool empty() const noexcept { return size_ == 0; }

  template <std::size_t Count>
  constexpr span<element_type, Count> first() const {
    return span<element_type, Count>(data_, Count);
  }
  constexpr span<element_type, dynamic_extent> first(size_type count) const {
    return span<element_type, dynamic_extent>(data_, count);
  }
  template <std::size_t Count>
  constexpr span<element_type, Count> last() const {
    return span<element_type, Count>(data_ + size_ - Count, Count);
  }
  constexpr span<element_type, dynamic_extent> last(size_type count) const {
    return span<element_type, dynamic_extent>(data_ + size_ - count, count);
  }
  constexpr span<element_type, dynamic_extent> subspan(
      size_type off, size_type count = dynamic_extent) const {
    return span<element_type, dynamic_extent>(
        data_ + off, count == dynamic_extent ? size_ - off : count);
  }

 private:
  pointer data_;
  size_type size_;
};

template <class T, std::size_t N> span(T (&)[N]) -> span<T, N>;
template <class T, std::size_t N> span(std::array<T, N>&) -> span<T, N>;
template <class T, std::size_t N> span(const std::array<T, N>&) -> span<const T, N>;
template <class C> span(C&) -> span<typename C::value_type>;
template <class C> span(const C&) -> span<const typename C::value_type>;

}  // namespace std
#endif  // C++17 and no real <span>
#endif  // K40_CXX17_COMPAT_H
