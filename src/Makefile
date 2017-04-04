# CXX=clang++ -std=c++14
CXX=g++
CXXFLAGS=-O3 -std=c++14
#CXXFLAGS=-O3 -Wall
LDFLAGS=-L/usr/lib64/atlas -ltatlas

sources = network.cpp \
	matrix.cpp \
	main.cpp \
	train.cpp \
	input.cpp \
        ../examples/iris.cpp

obj = $(sources:.cpp=.o)

headers = network.hpp \
	matrix.hpp \
	error.hpp \
	activation.hpp \
	train.hpp \
	input.hpp \
	utility.hpp \
        ../examples/examples.h

nn : $(obj)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

wc :
	wc -l $(sources) $(headers)


.PHONY : clean
clean :
	-rm *.o core.* core vgcore.*
	-rm *.dep
	-rm *~
	-rm nn

# generate dependencies
%.dep: %.cpp
	$(CXX) -MM $< | sed 's|$*\.o[ ]*|& $@|g' > $@

# include dependencies
ifneq ($(MAKECMDGOALS),clean)
include $(sources:.cpp=.dep)
endif
