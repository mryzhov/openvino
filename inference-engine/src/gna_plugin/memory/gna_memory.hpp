// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna_mem_requests.hpp"
#include <ie_memcpy.h>
#include "gna_mem_requests_queue.hpp"
#include <cstdint>
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <functional>
#include <iostream>
#include "gna_lib_ver_selector.hpp"
#include "gna_memory_solver.hpp"

// #ifdef GNA_HEAP_PROFILER
#include <iomanip>
// #endif

namespace GNAPluginNS {
namespace memory {
/**
 * @brief encapsulate various request to allocate GNA specific memory,
 * in order to issue single allocation call and configure actual pointers in requests
 * @tparam Allocator - a GNAAllocator in case of actual HW offloads
 */
template<class Allocator = std::allocator<uint8_t>>
class GNAMemory : public GNAMemRequestsQueue {
    std::vector<MemRequest> _future_heap;
    std::list<std::vector<char>> _local_storage;
    size_t _total = 0;
    size_t _rw_section_size = 0;
    size_t _ro_section_size = 0;
    Allocator _allocator;
    std::shared_ptr<uint8_t> heap = nullptr;
    size_t _page_alignment = 1;
    bool _is_optimized = false;

    class GNAMemRequestsReadOnlyQueue : public GNAMemRequestsQueue {
        std::reference_wrapper<GNAMemRequestsQueue> _that;
     public:
        explicit GNAMemRequestsReadOnlyQueue(GNAMemory & that) : _that(that) {
        }
        rRegion regionType() const override {
            return REGION_RO;
        };
        std::vector<MemRequest> & futureHeap()  override {
            return _that.get().futureHeap();
        }
        std::list<std::vector<char>> &localStorage() override {
            return _that.get().localStorage();
        }
    };

    GNAMemRequestsReadOnlyQueue readOnlyFrontEnd;

 public:
    explicit GNAMemory(size_t pageAlignment = 1)
        : readOnlyFrontEnd(*this), _page_alignment(pageAlignment) {}

    explicit GNAMemory(const Allocator &a, size_t pageAlignment = 1)
        : _allocator(a), readOnlyFrontEnd(*this), _page_alignment(pageAlignment) {}

    GNAMemRequestsQueue & readonly() {
        return readOnlyFrontEnd;
    }

    /**
     * @brief calculates size required for all requests, allocates memory and updates pointers
     */
    void commit() {
        // 1st stage -- looking for expandable bind requests:
        for (auto &originated : _future_heap) {
            if (originated._type == REQUEST_BIND) continue;
            size_t offset = 0;
            std::cout << "originated type: " << rTypeToStr(originated._type) << std::endl;
            std::cout << "originated ptr_in: " << originated._ptr_in << std::endl;
            std::cout << "originated otr_out: " << originated._ptr_out << std::endl;
            iterate_binded(originated, [&](MemRequest & reference, MemRequest & binded) {
                if (&originated == &reference) {
                    offset = 0;
                }
                offset += binded._offset;
                auto current = offset + ALIGN(binded._num_elements * binded._element_size, binded._alignment);
                auto original_no_pad = ALIGN(originated._num_elements * originated._element_size, originated._alignment);
                auto original_with_pad = ALIGN(originated._num_elements * originated._element_size + originated._padding, originated._alignment);

                originated._padding = ALIGN(std::max(original_with_pad, current), originated._alignment) - original_no_pad;
                originated._life_limits = std::make_pair(originated._execution_id, binded._execution_id);
                std::cout << "life limits updated: [" << originated._execution_id << ", " << binded._execution_id << "]" << std::endl;
            });
        }

        updateSectionsSizes();



        // std::vector<MemorySolver::Box> boxes;
        // std::cout << "Memory Requests:" << std::endl;
        // for (int i = 0; i < _future_heap.size(); i++) {
        //     if (_future_heap[i]._type & REQUEST_BIND || _future_heap[i]._region != REGION_RW) {
        //         continue;
        //     }

        //     // &box = boxes[i];
        //     auto original_with_pad =
        //         ALIGN(_future_heap[i]._num_elements * _future_heap[i]._element_size + _future_heap[i]._padding, _future_heap[i]._alignment);

        //     int start = std::get<0>(_future_heap[i]._life_limits);
        //     int stop = std::get<1>(_future_heap[i]._life_limits);
        //     // int64_t box_id = reinterpret_cast<int64_t>(_future_heap[i]._ptr_out);
        //     //MemorySolver::Box box = {start, stop, static_cast<int64_t>(original_with_pad), i};
        //     boxes.push_back({start, stop, static_cast<int64_t>(original_with_pad), i});
        // }
        // MemorySolver memSolver(boxes);
        // size_t total_size = memSolver.solve();
        // std::cout << "REQESTED_OPT size=" << total_size << "\n";

        solveMemory(memory::rRegion::REGION_RW);

        std::cout << "REQUESTED total size=" << _total << "\n";
        std::cout << "REQUESTED RO size=" << _ro_section_size << "\n";
        std::cout << "REQUESTED RW size=" << _rw_section_size << "\n";
        // allocation with memory setting to 0 internally

        heap = allocate(_total);
        auto setupOffsets = [&](std::function<bool(MemRequest & request)> filter, size_t offset) {
            size_t local_offset = offset;
            for (auto &re : _future_heap) {
                if (re._type == REQUEST_BIND) continue;
                if (filter(re)) continue;

                auto sz = re._element_size * re._num_elements;
                // ptrdiff_t pos = std::distance(_future_heap.begin(), std::find(_future_heap.begin(), _future_heap.end(), re));
                // std::cout << pos << std::endl;
                if (re._region == REGION_RW) {
                    local_offset = re._offset;
                }
                if (re._ptr_out != nullptr) {
                    auto cptr = heap.get() + local_offset;
                    size_t cptr_avail_size = _total - local_offset;
                    if (re._type & REQUEST_BIND) {
                        cptr = reinterpret_cast<uint8_t*>(*reinterpret_cast<void **>(re._ptr_out));
                        cptr_avail_size = sz;
                    } else {
                        *reinterpret_cast<void **>(re._ptr_out) = cptr;
                    }
                    std::cout << "ALLOCATED=" << static_cast<void*>(cptr) << ", size=" << re._element_size * re._num_elements << "\n";
                    iterate_binded(re, [](MemRequest & reference, MemRequest & binded) {
                        *reinterpret_cast<void **>(binded._ptr_out) =
                            binded._offset + reinterpret_cast<uint8_t *>(*reinterpret_cast<void **>(reference._ptr_out));
                        binded._num_elements = reference._num_elements;
                        binded._element_size = reference._element_size;
                    });

                    std::cout << "size=" << ALIGN(sz, re._alignment) << "\n" << std::flush;

                    switch (re._type & ~REQUEST_BIND) {
                        case REQUEST_ALLOCATE :
                            break;
                        case REQUEST_STORE : {
                            if (re._ptr_in != nullptr) {
                                ie_memcpy(cptr, cptr_avail_size, re._ptr_in, sz);
                            } else {
                                size_t of = 0;
                                for (int i = 0; i < re._num_elements; i++, of += re._element_size) {
                                    std::copy(std::begin(re._data), std::end(re._data), cptr + of);
                                }
                            }
                            break;
                        }
                        case REQUEST_INITIALIZER : {
                            re._initializer(cptr, sz);
                            break;
                        }
                    }
                }
                if (!(re._type & REQUEST_BIND)) {
                    if (re._region != REGION_RW) {
                         local_offset += ALIGN(sz + re._padding, re._alignment);
                    }
                    std::cout << "offset=" << local_offset << std::endl;
                }
            }
            //updateSectionsSizes();
        };



        setupOffsets([](GNAPluginNS::memory::MemRequest & request) {
            // TODO: consume bind requests separately from storage type
            return !(request._type & REQUEST_BIND) && (request._region != REGION_RW);
        }, 0);

        setupOffsets([](GNAPluginNS::memory::MemRequest & request) {
            return (request._type & REQUEST_BIND) || request._region != REGION_RO;
        }, _rw_section_size);
    }

    void *getBasePtr() {
        return heap.get();
    }

    size_t getRWBytes() {
        updateSectionsSizes();
        return _rw_section_size;
    }

    size_t getTotalBytes() {
        updateSectionsSizes();
        return _total;
    }

 protected:
    rRegion regionType() const override {
        return REGION_RW;
    };
    std::vector<MemRequest> & futureHeap()  override {
        return _future_heap;
    }
    std::list<std::vector<char>> &localStorage() override {
        return _local_storage;
    }

    template<class T>
    void iterate_binded(GNAPluginNS::memory::MemRequest & reference, const T & visitor) {
        for (auto &re : _future_heap) {
            if ((re._type & REQUEST_BIND) && (re._ptr_in == reference._ptr_out)) {
                std::cout << "  [binded=" << rTypeToStr(re._type) << ", ptr=" << re._ptr_out <<"]\n";
                visitor(reference, re);
                // primitive loop check
                if (re._ptr_in == re._ptr_out) continue;
                // TODO: no circular dependency checking, only tree-style dependency with loops supported
                iterate_binded(re, visitor);
            }
        }
    }


    std::shared_ptr<uint8_t> allocate(size_t bytes) {
        std::shared_ptr<uint8_t> sp(_allocator.allocate(bytes), [=](uint8_t *p) {
            _allocator.deallocate(p, bytes);
        });
        std::fill(sp.get(), sp.get() + bytes, 0);
        return sp;
    }

 protected:
    void solveMemory(GNAPluginNS::memory::rRegion regType) {
        switch (regType) {
            case REGION_RW: {
                    std::vector<MemorySolver::Box> boxes;
                    for (int i = 0; i < _future_heap.size(); i++) {
                        if (_future_heap[i]._type & REQUEST_BIND || _future_heap[i]._region != REGION_RW) {
                            continue;
                        }

                        auto original_with_pad = ALIGN(_future_heap[i]._num_elements * _future_heap[i]._element_size + _future_heap[i]._padding,
                                                    _future_heap[i]._alignment);
                        int start = std::get<0>(_future_heap[i]._life_limits);
                        int stop = std::get<1>(_future_heap[i]._life_limits);

                        boxes.push_back({start, stop, static_cast<int64_t>(original_with_pad), i});
                    }
                    MemorySolver memSolver(boxes);
                    _rw_section_size = memSolver.solve();

                    // setting offsets
                    for (auto box : boxes) {
                        _future_heap[box.id]._offset = memSolver.getOffset(box.id);
                    }
                    std::cout << "REQESTED_RW_OPT size=" << _rw_section_size << "\n";
                }
                break;

            default:
                break;
            }
    }

    void updateSectionsSizes() {
        // count total size and size of read/write regions
        _rw_section_size = 0;
        _ro_section_size = 0;
        for (auto &re : _future_heap) {
            auto current = ALIGN(re._num_elements * re._element_size + re._padding, re._alignment);
// #ifdef GNA_HEAP_PROFILER
            std::cout << ": " << " region: " << rRegionToStr(re._region) << ", " <<
                    "type: " << std::setw(20) << rTypeToStr(re._type) <<
                    std::setw(5) << re._num_elements <<
                    static_cast<int>(re._element_size) << ", " <<
                    re._padding << ", " <<
                    std::setw(3) << re._offset << ", " <<
                    std::setw(3) << re._alignment << ", " <<
                    "life_time: " << std::get<0>(re._life_limits) << ":" << std::get<1>(re._life_limits) << ", " <<
                    std::endl;
// #endif
            if (re._type == REQUEST_BIND) continue;

            if (re._region == REGION_RW) {
                _rw_section_size += current;
            } else {
                _ro_section_size += current;
            }
        }
// #ifdef GNA_HEAP_PROFILER
        std::cout << "ro_section_size: " << _ro_section_size << std::endl;
        std::cout << "rw_section_size: " << _rw_section_size << std::endl;
// #endif
        _rw_section_size = ALIGN(_rw_section_size, _page_alignment);
        _ro_section_size = ALIGN(_ro_section_size, _page_alignment);
        _total = _rw_section_size + _ro_section_size;
// #ifdef GNA_HEAP_PROFILER
        std::cout << "Aligned ro_section_size: " << _ro_section_size << std::endl;
        std::cout << "Aligned rw_section_size: " << _rw_section_size << std::endl;
// #endif
    }
};
}  // namespace memory
}  // namespace GNAPluginNS
