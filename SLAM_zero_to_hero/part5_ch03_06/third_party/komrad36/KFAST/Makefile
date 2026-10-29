EXECUTABLE_NAME=KFAST
CPP=g++
CC=gcc
INC=
CPPFLAGS=-Wall -Wextra -Werror -Wshadow -pedantic -Ofast -std=gnu++17 -fomit-frame-pointer -mavx2 -march=native -mfma -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize
CFLAGS=-Wall -Wextra -Werror -Wshadow -pedantic -Ofast -std=c11 -fomit-frame-pointer -mavx2 -march=native -mfma -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize
LIBS=-lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgcodecs -lpthread
CPPSOURCES=$(wildcard *.cpp)
CSOURCES=$(wildcard Rosten/*.c)

OBJECTS=$(CPPSOURCES:.cpp=.o) $(CSOURCES:.c=.o)

.PHONY : all
all: $(CPPSOURCES) $(CSOURCES) $(EXECUTABLE_NAME)

$(EXECUTABLE_NAME) : $(OBJECTS)
	$(CPP) $(CPPFLAGS) $(OBJECTS) $(PROFILE) -o $@ $(LIBS)

%.o:%.cpp
	$(CPP) -c $(INC) $(CPPFLAGS) $(PROFILE) $< -o $@

%.o:%.c
	$(CC) -c $(INC) $(CFLAGS) $(PROFILE) $< -o $@

.PHONY : clean
clean:
	rm -rf *.o Rosten/*.o $(EXECUTABLE_NAME)
