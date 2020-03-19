CC = gcc
CFLAGS = -g -Wall
LDFLAGS = -lpthread

TARGETS = pthread_create pthread_exit

all:$(TARGETS)

%:%.o
	$(CC) $(LDFLAGS) $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(TARGETS)
