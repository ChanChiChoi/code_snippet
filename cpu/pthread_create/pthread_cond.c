/*
无论哪种等待方式，都必须和一个互斥锁配合，以防止多个线程同时请求pthread_cond_wait()（或pthread_cond_timedwait()，下同）的竞争条件（Race Condition）。mutex互斥锁必须是普通锁（PTHREAD_MUTEX_TIMED_NP）或者适应锁（PTHREAD_MUTEX_ADAPTIVE_NP），且在调用pthread_cond_wait()前必须由本线程加锁（pthread_mutex_lock()），而在更新条件等待队列以前，mutex保持锁定状态，并在线程挂起进入等待前解锁。
在条件满足从而离开pthread_cond_wait()之前，mutex将被重新加锁，以与进入thread_cond_wait()前的加锁动作对应。

使用pthread_cond_wait方式如下：

>> pthread _mutex_lock(&mutex)
>> while或if(线程执行的条件是否成立)
>>       pthread_cond_wait(&cond, &mutex);
>> 线程执行
>> pthread_mutex_unlock(&mutex);
*/

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; /*初始化互斥锁*/
pthread_cond_t cond = PTHREAD_COND_INITIALIZER; //初始化条件变量

void *thread1(void *);
void *thread2(void *);

int i=1;
 
int main(void){
    pthread_t t_a;
    pthread_t t_b;

    pthread_create(&t_a,NULL,thread1,(void *)NULL);/*创建进程t_a*/
    pthread_create(&t_b,NULL,thread2,(void *)NULL); /*创建进程t_b*/

    pthread_join(t_b, NULL);/*等待进程t_b结束*/

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);

    return 0;
}
 
void *thread1(void *junk){

    for(i=1;i<=9;i++){

        printf("IN one, i=%d\n", i);
        pthread_mutex_lock(&mutex);//锁定
        if(i%3==0)
            pthread_cond_signal(&cond);/*条件改变，发送信号，通知t_b进程*/
        else
            printf("thead1:%d\n",i);
        pthread_mutex_unlock(&mutex);//*解锁互斥量*/
        printf("Up Mutex\n");
        sleep(3);
    }
    return 0;
}
 
void *thread2(void *junk){                                                   

    while(i<9){                                               

        printf("IN two, i=%d\n", i);                
        pthread_mutex_lock(&mutex);                 
        if(i%3!=0)                                  
            pthread_cond_wait(&cond,&mutex);/*等待*/
        printf("thread2:%d\n",i);                   
        pthread_mutex_unlock(&mutex);               
        printf("Down Mutex\n");                     
        sleep(3);                                   
    }                                               
    return 0;
}     
