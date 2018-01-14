SRIPATH = /Users/andy/srilm-1.5.10
MACHINE_TYPE = macosx
LM = bigram.lm

CXX = g++
CXXFLAGS = -O3 -I$(SRIPATH)/include
vpath lib%.a $(SRIPATH)/lib/$(MACHINE_TYPE)

TARGET = mydisambig
SRC = mydisambig.cpp
OBJ = $(SRC:.cpp=.o)

.PHONY: all clean

all: $(TARGET)

map:
	python3 map.py Big5-Zhuyin.map ZhuYin-Big5.map 

run:
	rm -rf result2
	mkdir result2
	./mydisambig -text testdata/1.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/1.txt
	./mydisambig -text testdata/2.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/2.txt
	./mydisambig -text testdata/3.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/3.txt
	./mydisambig -text testdata/4.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/4.txt
	./mydisambig -text testdata/5.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/5.txt
	./mydisambig -text testdata/6.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/6.txt
	./mydisambig -text testdata/7.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/7.txt
	./mydisambig -text testdata/8.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/8.txt
	./mydisambig -text testdata/9.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/9.txt
	./mydisambig -text testdata/10.txt -map ZhuYin-Big5.map -lm $(LM) -order 2 > result2/10.txt

$(TARGET): $(OBJ) -loolm -ldstruct -lmisc
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	$(RM) $(OBJ) $(TARGET)
