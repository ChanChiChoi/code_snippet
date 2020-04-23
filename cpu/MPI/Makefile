

all: hello


hello:hello.cpp
	mpic++ $^ -o $@

clean:
	rm -f hello

run:
	mpirun -np 4 ./hello
