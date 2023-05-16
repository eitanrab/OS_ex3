//
// Created by eitan.rab on 5/16/23.
//

#include <utility>
#include <vector>
#include <pthread.h>
#include <cstdio>
#include <atomic>
#include <algorithm>
#include <queue>
#include "Barrier/Barrier.h"
#include "MapReduceFramework.h"


//todo prnt error mesage whenever system call doesn't work

struct ThreadContext {
    int threadID;

    //first 31 bits for current position in vector
    //second 31 bits for counter of finished jobs
    //2 last bits for state flag
    std::atomic<uint64_t>* atomic_counter;
    Barrier* barrier;
    InputVec inputVec;
    IntermediateVec ** interVecs;
    std::vector<IntermediateVec*> ** interVecs1;
    OutputVec* outputVec;
    const MapReduceClient* client;
    int multiThreadLevel;
};

void
shuffle(IntermediateVec **intermediateVecs, ThreadContext *tc);

int compare (IntermediatePair a, IntermediatePair b){
    return *a.first < *b.first;
}

void map(ThreadContext *tc);

void reduce(ThreadContext *tc);

void* mapSortReduce(void* arg){
    auto tc = (ThreadContext*) arg;

    map(tc);
//    auto  interVec= new (std::nothrow) (*(tc->interVecs))[tc->threadID];
    auto interVec = *(tc->interVecs+tc->threadID);
    std::sort(interVec->begin(), interVec->end(), compare);

    tc->barrier->barrier();

    if(tc->threadID) {
        tc->barrier->barrier();
    }

    else {//if threadID==0 - shuffle
        (*(tc->atomic_counter)) = ((uint64_t)1)<<63 ;
        shuffle((tc->interVecs), tc);
        (*(tc->atomic_counter)) = (((uint64_t)1)<<63) + (((uint64_t)1)<<62);
        //todo make sure that we can 0 the counter here before passing the barrier
        tc->barrier->barrier();

    }

    //reduce
    reduce(tc);
    return nullptr;

}

void reduce(ThreadContext *tc) {
    while(true) {
        uint64_t old_value = (*(tc->atomic_counter))++;
        int val= old_value<<33>>33;
        printf("%d\n", (*(tc->interVecs1))->size());
        if (val < (*(tc->interVecs1))->size()-1) { //todo chek that size is actualy -1
            auto  keyVec= (**(tc->interVecs1))[val];
            tc->client->reduce(keyVec, tc);
        }
        else{
            break;
        }
    }
}

void map(ThreadContext *tc) {
    printf("got hre !!\n");
    fflush(stdout);
    (*(tc->atomic_counter)) |= ((uint64_t)1)<<62;
    printf("%lu\n", (*(tc->atomic_counter)).load());
    fflush(stdout);
    while(true) {
        uint64_t old_value = (*(tc->atomic_counter))++;
        int val= old_value<<33>>33;
        if (val < tc->inputVec.size()) {
            tc->client->map(tc->inputVec[old_value].first,
                            tc->inputVec[old_value].second,
                            tc);
        }
        else{
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
shuffle(IntermediateVec **intermediateVecs, ThreadContext *tc) {
    IntermediateVec** interVecs = intermediateVecs;
    auto tempVecs = new (std::nothrow) std::vector<IntermediateVec*>; //todo makee sure this doesn't need alloction
    auto maxKeys = new IntermediateVec (tc->multiThreadLevel);
    for (int i = 0; i < tc->multiThreadLevel; i++) {
        if((*(interVecs+i))->empty()){
            (*maxKeys)[i] = IntermediatePair (nullptr, nullptr);
        }
        else {
            (*maxKeys)[i] = (*(interVecs + i))->back();
        }
    }
    bool flag_working=true;
    while (flag_working) {
        // Find the maximum element and its iterator
        auto maxel=IntermediatePair (nullptr, nullptr);
        for(auto p: *maxKeys){
            if(maxel.first== nullptr){
                maxel=p;
            } else if(p.first != nullptr) {
                maxel = (*(p.first)) < (*(maxel.first)) ? maxel : p;
            }
        }
//        auto maxIter = std::max_element(maxKeys->begin(), maxKeys->end());
        auto temp = new (std::nothrow) IntermediateVec ;
        flag_working=false;
        for(int i=0; i<maxKeys->size(); i++){
            while(!(*(interVecs + i))->empty() && (*maxKeys)[i].first!= nullptr &&
            !(((*(*maxKeys)[i].first))<(*maxel.first) || (*maxel.first)<(*(*maxKeys)[i].first))){
                flag_working=true;
                temp->push_back((*maxKeys)[i]);
                (*(interVecs + i))->pop_back();
                if((*(interVecs + i))->empty()){
                    (*maxKeys)[i]=IntermediatePair (nullptr, nullptr);
                }
                else {
                    (*maxKeys)[i] = (*(interVecs + i))->back();
                }
                (*(tc->atomic_counter)) += 1 << 31;

            }
        }
        tempVecs->push_back(temp);
    }
    (*(tc->interVecs1))=tempVecs;
    delete maxKeys;

}



struct JobContext {
    pthread_t* threads;
    ThreadContext** contexts;
    Barrier * barrier;
    bool waiting = false;
    int multiThreadLevel;


    JobContext(pthread_t*  threadVec,
               ThreadContext ** contextVec,
                Barrier* barrierObj, int multithreadLevel)
            : threads(threadVec),
              contexts(contextVec),
              barrier(barrierObj),
              multiThreadLevel(multiThreadLevel){}
};





JobHandle
startMapReduceJob(const MapReduceClient &client, const InputVec &inputVec, OutputVec &outputVec, int multiThreadLevel) {
    auto threads= new (std::nothrow) pthread_t[multiThreadLevel];
    auto contexts = new (std::nothrow) ThreadContext*[multiThreadLevel];
    auto barrier = new (std::nothrow) Barrier(multiThreadLevel);
    auto atomic_counter = new (std::nothrow) std::atomic<uint64_t>(0);
    auto intermediateVecs = new (std::nothrow) IntermediateVec*[multiThreadLevel];
    auto interVecs1 = new (std::nothrow) std::vector<IntermediateVec*> *;
    for (int i=0; i < multiThreadLevel; i++) {
        intermediateVecs[i] = new (std::nothrow) IntermediateVec;
    }
    for (int i = 0; i < multiThreadLevel; ++i) {
        contexts[i] = new (std::nothrow) ThreadContext{i, atomic_counter, barrier, inputVec, intermediateVecs, interVecs1,
                       &outputVec, &client, multiThreadLevel};
    }

    for (int i = 0; i < multiThreadLevel; ++i) {
        pthread_create(threads+i, nullptr, mapSortReduce, *(contexts+i));
    }

    // Initialize the members of JobContext as needed

    JobContext* jobContext = new JobContext(threads, contexts, barrier, multiThreadLevel);



    // Cast JobContext* to JobHandle
    JobHandle jobHandle = static_cast<JobHandle>(jobContext);

    return jobHandle;
}

void emit2(K2 *key, V2 *value, void *context) {
    auto* tc = (ThreadContext*) context;
    auto interVer = tc->interVecs;
    interVer[tc->threadID]->push_back(IntermediatePair(key, value));
    (*(tc->atomic_counter)) += 1 << 31;
}

void emit3(K3 *key, V3 *value, void *context) {
    auto* tc = (ThreadContext*) context;
    tc->outputVec->push_back(OutputPair(key, value));
    (*(tc->atomic_counter)) += 1 << 31;
}

void waitForJob(JobHandle job) { //TODO: add mutex
    auto* jc = (JobContext*) job;
    if (jc->waiting) {
        return;
    }
    for (int i=0; i<jc->multiThreadLevel; i++) {
        pthread_join(jc->threads[i], nullptr);
    }
    jc->waiting = true;
}

void getJobState(JobHandle job, JobState* state) {
    auto* jc = (JobContext*) job;
    auto atomic_counter_val = (*(jc->contexts))[0].atomic_counter->load();
    state->stage = stage_t(atomic_counter_val>>62);
    state->percentage = 100 * (((float) (*(jc->contexts))[0].inputVec.size()) / (atomic_counter_val << 2 >>31));
}

void closeJobHandle(JobHandle job){
    waitForJob(job);
//    auto jc = (JobContext*) job;
//    for (int i=0; i<jc->multiThreadLevel; i++) {
//        delete jc->contexts[i]->interVecs[i];
//    }
//    delete jc->contexts[0]->interVecs1;
//    delete (*(jc->contexts))[0].atomic_counter;
//    delete[] (*(jc->contexts))[0].interVecs;
//    for (int i=0; i<jc->multiThreadLevel; i++) {
//        delete jc->contexts[i];
//    }
//    delete jc->contexts;
//    delete[] jc->threads;
//    delete jc->barrier;
//    delete jc;
}