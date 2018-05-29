CCOMP ?= icpc
CFLAGS ?= -qopenmp -O3 -g -qopt-report=5

all:
	$(CCOMP) stream.c $(CFLAGS) -S
	$(CCOMP) stream.c $(CFLAGS) -o STREAM.x

clean:
	rm -f *.s *.o *.optrpt STREAM.x
