CC = gcc
CFLAGS = -g -Wall
LDFLAGS = -lpthread

OBJECT = pthread_create pthread_exit

all:$(OBJECT)


$(OBJECT):$(addsuffix .o,$(OBJECT))
	$(CC) $(LDFLAGS) $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
	rm -f $(OBJECT)
