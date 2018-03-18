#sigverse header
SIG_SRC  = $(SIGVERSE_PATH)/include/sigverse

all: $(OBJS)

#compile
./%.so: ./%.cpp
	g++ -DCONTROLLER -DNDEBUG -DUSE_ODE -DdDOUBLE -I$(SIG_SRC) -I$(SIG_SRC)/comm/controller -fPIC -shared -o $@ $< `pkg-config --cflags --libs opencv`

clean:
	rm ./*.so
