CXXFLAGS= -Wall -Wextra -pedantic -std=c99 -O2 -fomit-frame-pointer
#override CXXFLAGS+= -Wall -fsigned-char

LIBS=-lglfw -lGL -lm
OBJS= \
	demo.o \
	texture.o
GPU_OBJS= \
	gpu-demo.o \
	texture.o

default: all

all: demo gpu-demo

clean:
	-$(RM) $(OBJS) $(GPU_OBJS) demo gpu-demo

demo: $(OBJS)
	$(CC) $(CXXFLAGS) -o demo $(OBJS) $(LIBS)

gpu-demo: $(GPU_OBJS)
	$(CC) $(CXXFLAGS) -o gpu-demo $(GPU_OBJS) $(LIBS)

%.o: %.c
	$(CC) $(CXXFLAGS) -c -o $@ $<