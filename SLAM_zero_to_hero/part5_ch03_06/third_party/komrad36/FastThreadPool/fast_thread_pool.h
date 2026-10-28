/*******************************************************************
*
*    Author: Kareem Omar
*    kareem.h.omar@gmail.com
*    https://github.com/komrad36
*
*    Last updated Oct 18, 2021
*******************************************************************/

#pragma once

#include <atomic>
#include <cstdint>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

class FastThreadPool
{
private:
    struct FastThreadPoolItem
    {
        std::atomic<FastThreadPoolItem*> m_next;
        void (*m_f)();
    };

    static void Pause()
    {
#ifdef __SSE2__
        _mm_pause();
#endif
    }

public:
    FastThreadPool()
    {
        // S1
        m_pHead.store(nullptr, std::memory_order_relaxed);
        m_pTail.store(nullptr, std::memory_order_relaxed);
    }

    void AddWork(void f())
    {
        FastThreadPoolItem* pItem = new FastThreadPoolItem;

        // S2
        pItem->m_next.store(nullptr, std::memory_order_relaxed);
        pItem->m_f = f;

        FastThreadPoolItem* pTail = m_pTail.load(std::memory_order_relaxed);

        do
        {
            while ((uintptr_t)pTail & 1)
            {
                pTail = m_pTail.load(std::memory_order_relaxed);
                Pause();
            }

            // L1
            //
            // S3
            // HAPPENS-AFTER: S2
            // SYNC-WITH: L3
        } while (!m_pTail.compare_exchange_weak(
            pTail,
            pTail ? (FastThreadPoolItem*)((uintptr_t)pTail | 1) : pItem,
            std::memory_order_release,
            std::memory_order_relaxed));

        if (pTail)
        {
            // S4
            // HAPPENS-AFTER: S2, S3
            // SYNC-WITH: L3, L5
            pTail->m_next.store(pItem, std::memory_order_release);

            // S5
            // HAPPENS-AFTER: S4
            // SYNC-WITH: L3
            m_pTail.store(pItem, std::memory_order_release);
        }
        else
        {
            FastThreadPoolItem* pHead = m_pHead.load(std::memory_order_relaxed);

            do
            {
                while (pHead)
                {
                    pHead = m_pHead.load(std::memory_order_relaxed);
                    Pause();
                }

                // L2
                //
                // S6
                // HAPPENS-AFTER: S2, S3
                // SYNC-WITH: L3
            } while (!m_pHead.compare_exchange_weak(pHead, pItem, std::memory_order_release, std::memory_order_relaxed));
        }
    }

    void (*RemoveWork(void))()
    {
        // L3
        // HAPPENS-BEFORE: L4, L5
        FastThreadPoolItem* pHead = m_pHead.load(std::memory_order_relaxed);

        do
        {
            while (pHead == (FastThreadPoolItem*)(1))
            {
                pHead = m_pHead.load(std::memory_order_relaxed);
                Pause();
            }

            if (!pHead)
            {
                break;
            }

            // also L3
            // HAPPENS-BEFORE: L4, L5
            //
            // S7
        } while (!m_pHead.compare_exchange_weak(pHead, (FastThreadPoolItem*)(1), std::memory_order_acquire, std::memory_order_acquire));

        if (pHead)
        {
            void (*f)() = pHead->m_f;

            // L4
            FastThreadPoolItem* pTail = m_pTail.load(std::memory_order_relaxed);

            do
            {
                while ((uintptr_t)pTail == ((uintptr_t)pHead | 1))
                {
                    pTail = m_pTail.load(std::memory_order_relaxed);
                    Pause();
                }

                if (pTail != pHead)
                {
                    break;
                }

                // also L4
                // S8
            } while (!m_pTail.compare_exchange_weak(pTail, nullptr, std::memory_order_relaxed, std::memory_order_relaxed));

            // L5
            // HAPPENS-BEFORE:L6
            FastThreadPoolItem* pNext = pTail == pHead ? nullptr : pHead->m_next.load(std::memory_order_acquire);

            // S9
            m_pHead.store(pNext, std::memory_order_relaxed);

            // L6
            delete pHead;

            return f;
        }

        return nullptr;
    }

    ~FastThreadPool()
    {
        FastThreadPoolItem* pItem = m_pHead.load(std::memory_order_relaxed);
        while (pItem)
        {
            FastThreadPoolItem* pPrev = pItem;
            pItem = pItem->m_next.load(std::memory_order_relaxed);
            delete pPrev;
        }
    }

private:
    alignas(64) std::atomic<FastThreadPoolItem*> m_pHead;
    alignas(64) std::atomic<FastThreadPoolItem*> m_pTail;
};
