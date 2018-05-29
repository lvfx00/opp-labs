#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>

#define TASK_NUM 100
#define MAX_TIME 3000000

int rank;
int size;
int tasks_left;
int *tasks;
pthread_mutex_t mutex;
pthread_cond_t cond_worker, cond_asker;

void *worker(void *par) {
    int curr_task;
    while (1) {
        pthread_mutex_lock(&mutex);
        // take one task and process it
        if (tasks_left != 0) {
            --tasks_left;
            curr_task = tasks[tasks_left];
//            printf ("proc: %d, doing task with weight: %d. Tasks remain: %d \n", rank, curr_task, tasks_left);
            pthread_mutex_unlock(&mutex);
            usleep(curr_task); // TODO implement work
        } else {
            // signal asker to ask new tasks from another process
            pthread_cond_signal(&cond_asker);
            // wait for answer
            pthread_cond_wait(&cond_worker, &mutex);
            if (tasks_left == 0) {
                pthread_mutex_unlock(&mutex);
                break;
            }
            pthread_mutex_unlock(&mutex);
        }
    }
    printf("proc: %d, worker ended his work\n", rank);
}

void *taskAsker(void *par) {
    int ended = -1;
    int task = 0;
    while (1) {
        pthread_mutex_lock(&mutex);
        if (tasks_left == 0) {
            pthread_mutex_unlock(&mutex);

            for (int i = 0; i < size; i++) {
                if (i != rank) {
                    MPI_Send(&rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//                    printf ("proc: %d, sended request to %d\n", rank, i);
                    MPI_Recv(&task, 1, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // i-th process have task to share and returned it
                    if (task != -1) {
                        pthread_mutex_lock(&mutex);
                        tasks[0] = task;
                        tasks_left++;
                        printf ("proc: %d, recived task from %d. Tasks remain:%d\n", rank, i, tasks_left);
                        pthread_mutex_unlock(&mutex);
                        break;
                    }
                }
            }
            pthread_cond_signal(&cond_worker);
            if (task == -1) {
                for (int i = 0; i < size; i++) {
                    if (i != rank) {
                        MPI_Send(&ended, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    }
                }
                printf("proc: %d, taskAsker ended his work\n", rank);
                break;
            }
        } else {
            pthread_cond_wait(&cond_asker, &mutex);
            pthread_mutex_unlock(&mutex);
        }
    }
}

void *taskGiver(void *par) {
    int destrank;
    int ended_proc_num = 0;
    int no_more_tasks = -1;
    int task;
    MPI_Status test;
    while (1) {
        MPI_Recv(&destrank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &test);
        if (destrank == -1) {
            ended_proc_num++;
            // there is no more processes that may require a task
            if (ended_proc_num == size - 1) {
                printf("proc: %d, taskGiver ended his work\n", rank);
                break;
            }
        } else {
            pthread_mutex_lock(&mutex);
            if (tasks_left != 0) {
                tasks_left--;
                task = tasks[tasks_left];
                pthread_mutex_unlock(&mutex);
//                printf ("proc: %d, sending task to %d. Tasks remain: %d\n", rank, destrank, tasks_left);
                MPI_Send(&task, 1, MPI_INT, destrank, 2, MPI_COMM_WORLD);
            } else {
                pthread_mutex_unlock(&mutex);
//                printf ("proc: %d, sended to %d that he have no tasks\n", rank, destrank);
                MPI_Send(&no_more_tasks, 1, MPI_INT, destrank, 2, MPI_COMM_WORLD);
            }
        }
    }
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *all_tasks;
    int *q = calloc(sizeof(int), size); // number of tasks for each process
    assert(q != NULL);
    int *displs = calloc(sizeof(int), size); // displacements array
    assert(displs != NULL);

    for (int i = 0; i < size; ++i) {
        q[i] = i < TASK_NUM - size * (TASK_NUM / size) ? TASK_NUM / size + 1 : TASK_NUM / size;
        if (i != 0) {
            displs[i] = q[i - 1] + displs[i - 1];
        }
    }

    tasks = malloc(sizeof(int) * q[rank]); // tasks "weight" array
    assert(tasks != NULL);

    if (rank == 0) {
        srand(time(NULL));

        all_tasks = malloc(sizeof(int) * TASK_NUM);
        assert(all_tasks != NULL);

        for (int i = 0; i < TASK_NUM; i++) {
            // TODO implement meaningful task weight
            all_tasks[i] = rand() % MAX_TIME + 1;
        }
    }

    MPI_Scatterv(all_tasks, q, displs, MPI_INT, tasks, q[rank], MPI_INT, 0, MPI_COMM_WORLD);
    tasks_left = q[rank];

    if (rank == 0) {
        free(all_tasks);
    }
    free(q);
    free(displs);

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond_worker, NULL);
    pthread_cond_init(&cond_asker, NULL);
    pthread_t worker_thread, task_giver_thread, task_asker_thread;
    pthread_create(&task_asker_thread, NULL, taskAsker, NULL);
    pthread_create(&task_giver_thread, NULL, taskGiver, NULL);
    pthread_create(&worker_thread, NULL, worker, NULL);
    pthread_cond_signal(&cond_asker);
    pthread_cond_signal(&cond_worker);
    pthread_join(worker_thread, NULL);
    pthread_join(task_giver_thread, NULL);
    pthread_join(task_asker_thread, NULL);
    free(tasks);
//    pthread_cond_destroy(&cond_asker);
//    pthread_cond_destroy(&cond_worker);
    MPI_Finalize();
    return 0;
}

                     


