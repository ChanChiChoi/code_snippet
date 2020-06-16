CC = g++
LD_LIBRARYS = -lopencv_core -lopencv_highgui -lopencv_imgproc


OBJ = read_image

all:${OBJ}


${OBJ}:%:%.cpp
	${CC} ${LD_LIBRARYS} $^ -o $@


run:
	./${OBJ}

clean:
	rm -f *.o
	rm -f ${OBJ}
