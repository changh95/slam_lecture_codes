EXECUTABLE_NAME=RGB2Y
CPP=g++
INC=
CPPFLAGS=-Wall -Wextra -Werror -Wshadow -pedantic -Ofast -std=gnu++17 -fomit-frame-pointer -mavx2 -march=native -mfma -flto -funroll-all-loops -fpeel-loops -ftracer -ftree-vectorize
LIBS=-lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_highgui -lopencv_imgcodecs -lpthread
CPPSOURCES=$(wildcard *.cpp)

.PHONY : all
all: $(CPPSOURCES) $(EXECUTABLE_NAME)

$(EXECUTABLE_NAME) : $(CPPSOURCES)
	$(CPP) $(CPPFLAGS) $(CPPSOURCES) $(PROFILE) -o $@ $(LIBS)

.PHONY : clean
clean:
	rm -rf $(EXECUTABLE_NAME)
