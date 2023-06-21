CC = mpicc

SRC  = chrono.c
MAIN = gpt

CFLAGS = -I./
LDLIBS = -lmpi

all: $(SRC) $(MAIN)

clean:
	@ rm -f $(MAIN)

.PHONY: all clean