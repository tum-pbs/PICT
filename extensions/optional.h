#pragma once

#ifndef _INCLUDE_OPTIONAL_TYPE
#define _INCLUDE_OPTIONAL_TYPE

#include <torch/extension.h>

template <typename T>
using optional = c10::optional<T>;
const auto nullopt = c10::nullopt;

#endif //_INCLUDE_OPTIONAL_TYPE
