CC = gcc
CFLAGS = -g -Wall
LDFLAGS = -lpthread

main:pthread_create.o
	$(CC) $(LDFLAGS) $^ -o $@

pthread_create.o: pthread_create.c
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	rm -f pthread_create.o
	rm -f main
