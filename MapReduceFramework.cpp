//
// Created by eitan.rab on 5/16/23.
//

#include <vector>
#include <pthread.h>
#include <cstdio>
#include <algorithm>
#include <queue>
#include <iostream>

#include "Barrier/Barrier.h"
#include "MapReduceFramework.h"

#define SYS_ERROR "system error: "

struct ThreadContext
{
  int threadID;

  //first 31 bits for current position in vector
  //second 31 bits for counter of finished jobs
  //2 last bits for state flag
  std::atomic<uint64_t> *atomic_counter;
  std::atomic<uint64_t> *atomic_inter_counter;
  std::atomic<uint64_t> *atomic_outer_counter;
  Barrier *barrier;
  InputVec inputVec;
  IntermediateVec **interVecs;
  std::vector<IntermediateVec *> **interVecs1;
  OutputVec *outputVec;
  const MapReduceClient *client;
  int multiThreadLevel;
  pthread_mutex_t *mutex;
};

void
shuffle (IntermediateVec **intermediateVecs, ThreadContext *tc);

int compare (IntermediatePair a, IntermediatePair b)
{
  return *a . first < *b . first;
}

void map (ThreadContext *tc);

void reduce (ThreadContext *tc);

void unlockMutex (ThreadContext *tc);
void lockMutex (ThreadContext *tc);
void shuffle (ThreadContext *tc);
void *mapSortReduce (void *arg)
{
  auto tc = (ThreadContext *) arg;

  map (tc);
  auto interVec = *(tc -> interVecs + tc -> threadID);
  std::sort (interVec -> begin (), interVec -> end (), compare);

  tc -> barrier -> barrier ();

  shuffle (tc);

  //reduce
  reduce (tc);
  return nullptr;

}
void shuffle (ThreadContext *tc)
{
  if (tc -> threadID)
  {
    tc -> barrier -> barrier ();
  }
  else
  {//if threadID==0 - shuffle
    (*(tc -> atomic_counter)) = ((uint64_t) 1) << 63;
    shuffle ((tc -> interVecs), tc);

    (*(tc -> atomic_counter)) =
        (((uint64_t) 1) << 63) + (((uint64_t) 1) << 62);
    tc -> barrier -> barrier ();

  }
}

void reduce (ThreadContext *tc)
{
  while (true)
  {
    uint64_t old_value = (*(tc -> atomic_counter))++;
    long val = old_value << 33 >> 33;
    if (val < (*(tc -> interVecs1)) -> size ())
    {
      auto keyVec = (**(tc -> interVecs1))[val];
      tc -> client -> reduce (keyVec, tc);
    }
    else
    {
      break;
    }
  }
}
void lockMutex (ThreadContext *tc)
{
  if (pthread_mutex_lock (tc -> mutex) != 0)
  {
    std::cout << SYS_ERROR << "error in pthread_mutex_lock" << std::endl;
    exit (1);
  }
}
void unlockMutex (ThreadContext *tc)
{
  if (pthread_mutex_unlock (tc -> mutex) != 0)
  {
    std::cout << SYS_ERROR << "error on pthread_mutex_unlock" << std::endl;
    exit (1);
  }
}

void map (ThreadContext *tc)
{
  lockMutex (tc);
  (*(tc -> atomic_counter)) . fetch_or (((uint64_t) 1) << 62);
  unlockMutex (tc);

  while (true)
  {
    uint64_t old_value = (*(tc -> atomic_counter))++;

    int val = old_value << 33 >> 33;
    if (val < tc -> inputVec . size ())
    {
      lockMutex (tc);
      tc -> client -> map (tc -> inputVec[val] . first,
                           tc -> inputVec[val] . second,
                           tc);
      unlockMutex (tc);
      (*(tc -> atomic_counter)) += ((uint64_t) 1) << 31;

    }
    else
    {
      break;
    }
  }
}

/**
 *
 * @param IntermediateVecs input ascending order vecs
 * creates a new intermediateVec and replaces the current inter with the res inter
 */
void
shuffle (IntermediateVec **intermediateVecs, ThreadContext *tc)
{
  IntermediateVec **interVecs = intermediateVecs;
  auto tempVecs = new (std::nothrow) std::vector<IntermediateVec *>;
  if (!tempVecs)
  {
    std::cout << SYS_ERROR << MEMORY_ALLOCATION_ERR_MSSG << std::endl;
    exit (1);
  }
  auto maxKeys = new (std::nothrow) IntermediateVec (tc -> multiThreadLevel);
  if (!maxKeys)
  {
    std::cout << SYS_ERROR << MEMORY_ALLOCATION_ERR_MSSG << std::endl;
    exit (1);
  }
  for (int i = 0; i < tc -> multiThreadLevel; i++)
  {
    if ((*(interVecs + i)) -> empty ())
    {
      (*maxKeys)[i] = IntermediatePair (nullptr, nullptr);
    }
    else
    {
      (*maxKeys)[i] = (*(interVecs + i)) -> back ();
    }
  }
  bool flag_working = true;
  while (flag_working)
  {
    // Find the maximum element and its iterator
    auto maxel = IntermediatePair (nullptr, nullptr);
    for (auto p : *maxKeys)
    {
      if (maxel . first == nullptr)
      {
        maxel = p;
      }
      else if (p . first != nullptr)
      {
        maxel = (*(p . first)) < (*(maxel . first)) ? maxel : p;
      }
    }
    auto temp = new (std::nothrow) IntermediateVec;
    if (!temp)
    {
      std::cout << SYS_ERROR << MEMORY_ALLOCATION_ERR_MSSG << std::endl;
      exit (1);
    }
    flag_working = false;
    for (int i = 0; i < maxKeys -> size (); i++)
    {
      while (!(*(interVecs + i)) -> empty ()
             && (*maxKeys)[i] . first != nullptr &&
             !(((*(*maxKeys)[i] . first)) < (*maxel . first)
               || (*maxel . first) < (*(*maxKeys)[i] . first)))
      {
        flag_working = true;
        temp -> push_back ((*maxKeys)[i]);
        (*(interVecs + i)) -> pop_back ();
        if ((*(interVecs + i)) -> empty ())
        {
          (*maxKeys)[i] = IntermediatePair (nullptr, nullptr);
        }
        else
        {
          (*maxKeys)[i] = (*(interVecs + i)) -> back ();
        }
        (*(tc -> atomic_counter)) += ((uint64_t) 1) << 31;

      }
    }
    if (flag_working)
    {
      tempVecs -> push_back (temp);
    }
    else
    {
      delete temp;
    }
  }
  (*(tc -> interVecs1)) = tempVecs;
  delete maxKeys;

}

struct JobContext
{
  pthread_t *threads;
  ThreadContext **contexts;
  Barrier *barrier;
  bool waiting = false;
  int multiThreadLevel;
  pthread_mutex_t *mutex_wait_for_job;

  JobContext (pthread_t *threadVec,
              ThreadContext **contextVec,
              Barrier *barrierObj, int multithreadLevel,
              pthread_mutex_t *mutex)
      : threads (threadVec),
        contexts (contextVec),
        barrier (barrierObj),
        multiThreadLevel (multithreadLevel),
        mutex_wait_for_job (mutex)
  {}
};

JobHandle
startMapReduceJob (const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec, int multiThreadLevel)
{
  auto threads = new (std::nothrow) pthread_t[multiThreadLevel];
  auto contexts = new (std::nothrow) ThreadContext *[multiThreadLevel];
  auto barrier = new (std::nothrow) Barrier (multiThreadLevel);
  auto atomic_counter = new (std::nothrow) std::atomic<uint64_t> (0);
  auto atomic_inter_counter = new (std::nothrow) std::atomic<uint64_t> (0);
  auto atomic_out_counter = new (std::nothrow) std::atomic<uint64_t> (0);
  auto intermediateVecs = new (std::nothrow) IntermediateVec *[multiThreadLevel];
  auto interVecs1 = new (std::nothrow) std::vector<IntermediateVec *> *;
  auto mutex = new (std::nothrow)pthread_mutex_t (PTHREAD_MUTEX_INITIALIZER);
  auto mutex_wait_for_job = new (std::nothrow)pthread_mutex_t (PTHREAD_MUTEX_INITIALIZER);
  if (!threads || !contexts || !barrier || !atomic_counter
      || !atomic_inter_counter
      || !atomic_out_counter || !intermediateVecs || !interVecs1 || !mutex)
  {
    std::cout << SYS_ERROR << MEMORY_ALLOCATION_ERR_MSSG << std::endl;
    exit (1);
  }
  for (int i = 0; i < multiThreadLevel; i++)
  {
    intermediateVecs[i] = new (std::nothrow) IntermediateVec;
    if (!intermediateVecs[i])
    {
      std::cout << SYS_ERROR << MEMORY_ALLOCATION_ERR_MSSG << std::endl;
      exit (1);
    }
  }
  for (int i = 0; i < multiThreadLevel; ++i)
  {
    contexts[i] = new (std::nothrow) ThreadContext{i, atomic_counter,
                                                   atomic_inter_counter,
                                                   atomic_out_counter, barrier,
                                                   inputVec, intermediateVecs,
                                                   interVecs1,
                                                   &outputVec, &client,
                                                   multiThreadLevel, mutex};
    if (!contexts[i])
    {
      std::cout << SYS_ERROR << MEMORY_ALLOCATION_ERR_MSSG << std::endl;
      exit (1);
    }
  }

  for (int i = 0; i < multiThreadLevel; ++i)
  {
    if (pthread_create (threads + i, nullptr, mapSortReduce, *(contexts + i))
        != 0)
    {
      std::cerr << "system error: Failed to create threads." << std::endl;
      exit (1);
    }
  }

  // Initialize the members of JobContext as needed
  JobContext *jobContext = new (std::nothrow) JobContext (threads, contexts, barrier,
                                                          multiThreadLevel,
                                                          mutex_wait_for_job);
  if (!jobContext)
  {
    std::cout << "system error: out of memory" << std::endl;
    exit (1);
  }

  // Cast JobContext* to JobHandle
  auto jobHandle = static_cast<JobHandle>(jobContext);

  return jobHandle;
}

void emit2 (K2 *key, V2 *value, void *context)
{
  auto *tc = (ThreadContext *) context;
  auto interVer = tc -> interVecs;
  interVer[tc -> threadID] -> push_back (IntermediatePair (key, value));
  (*(tc -> atomic_inter_counter)) += 1;
}

void emit3 (K3 *key, V3 *value, void *context)
{
  auto *tc = (ThreadContext *) context;
  lockMutex (tc);
  tc -> outputVec -> push_back (OutputPair (key, value));
  unlockMutex (tc);
  (*(tc -> atomic_counter)) += ((uint64_t) 1) << 31;
}

void waitForJob (JobHandle job)
{ //TODO: add mutex
  auto *jc = (JobContext *) job;
  if (jc -> waiting)
  {
    return;
  }
  for (int i = 0; i < jc -> multiThreadLevel; i++)
  {
    if (pthread_join (jc -> threads[i], nullptr))
    {
      std::cout << SYS_ERROR << "error in pthread_join." << std::endl;
    }
  }
  jc -> waiting = true;
}

void getJobState (JobHandle job, JobState *state)
{
  auto *jc = (JobContext *) job;
  auto tc = (*(jc -> contexts))[0];
  auto atomic_counter_val = (*(jc -> contexts))[0] . atomic_counter -> load ();
  state -> stage = stage_t (atomic_counter_val >> 62);
  if (state -> stage == UNDEFINED_STAGE)
  {
    state -> percentage = 0;
    return;
  }
  unsigned long denom;
  if (state -> stage == MAP_STAGE)
  {
    denom = (*(jc -> contexts))[0] . inputVec . size ();
  }
  else if (state -> stage == SHUFFLE_STAGE)
  {
    denom = tc . atomic_inter_counter -> load ();
  }
  else //if (state -> stage == REDUCE_STAGE)
  {
    denom = (*(tc . interVecs1)) -> size ();
  }
  state -> percentage = 100 * (((float) (atomic_counter_val << 2 >> 33))
                               / ((float) denom));
}

void closeJobHandle (JobHandle job)
{
  waitForJob (job);
  auto jc = (JobContext *) job;
  for (int i = 0; i < jc -> multiThreadLevel; i++)
  {
    delete jc -> contexts[0] -> interVecs[i];
  }
  for (auto vec : **(jc -> contexts[0] -> interVecs1))
  {
    delete vec;
  }
  delete *(jc -> contexts[0] -> interVecs1);
  delete jc -> contexts[0] -> interVecs1;
  delete (*(jc -> contexts))[0] . atomic_counter;
  delete (*(jc -> contexts))[0] . atomic_inter_counter;
  delete (*(jc -> contexts))[0] . atomic_outer_counter;
  if (pthread_mutex_destroy ((*(jc -> contexts))[0] . mutex) != 0)
  {
    std::cout << SYS_ERROR << "error on mutex_destroy." << std::endl;
    exit (1);
  }
  delete (*(jc -> contexts))[0] . mutex;
  delete[] (*(jc -> contexts))[0] . interVecs;
  for (int i = 0; i < jc -> multiThreadLevel; i++)
  {
    delete jc -> contexts[i];
  }
  delete[] jc -> contexts;
  delete[] jc -> threads;
  delete jc -> barrier;
  if (pthread_mutex_destroy (jc -> mutex_wait_for_job) != 0)
  {
    std::cout << SYS_ERROR << "error on mutex_destroy." << std::endl;
    exit (1);
  }
  delete jc -> mutex_wait_for_job;
  delete jc;
}