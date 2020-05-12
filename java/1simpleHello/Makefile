JAVAC = javac
JAVA = java

OBJ = HelloJava

all:${addsuffix .class,${OBJ}}

HelloJava.class:HelloJava.java
	${JAVAC} $^ -o $@


run:
	@${JAVA} ${OBJ}


clean:
	rm -f *.class
