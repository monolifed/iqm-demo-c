CFLAGS= -Wall -Wextra -pedantic -std=c99 -O2

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
	-$(RM) *.o demo gpu-demo

demo: $(OBJS)
	$(CC) $(CFLAGS) -o demo $(OBJS) $(LIBS)

gpu-demo: $(GPU_OBJS)
	$(CC) $(CFLAGS) -o gpu-demo $(GPU_OBJS) $(LIBS)

gpu-demo.o: gpu-demo.c common.h iqm.h
	$(CC) $(CFLAGS) -c -o $@ $<

demo.o: demo.c common.h iqm.h
	$(CC) $(CFLAGS) -c -o $@ $<

texture.o: texture.c scale.h
	$(CC) $(CFLAGS) -c -o $@ $<