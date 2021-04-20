#---------------------------------------------------------------
# This makefile is set for static compile in your workspace.
# User setting
#---------------------------------------------------------------
#---------------------------Executable file name
TARGET   = test
#---------------------------Linking flags
LDFLAGS  = -lm -fopenmp -lstdc++fs
#---------------------------Include file
INCLUDES = -I ../../../vendor/
#---------------------------No make directory
NOMAKEDIR= .git% data% doc% src/bin%
#---------------------------object directory
OBJDIR = objs
#---------------------------path to src's objs
ICARATPATH =../../../src/objs
#---------------------------compiler
GCC = g++
#---------------------------for debug option
CFLAGS = -std=c++11 -MMD -MP -Wall -Wextra -g -fopenmp
#---------------------------for production option
#CFLAGS = -MMD -MP -O3 -fopenmp

#---------------------------------------------------------------
# Don't change the following
#---------------------------------------------------------------
#Run through the source directory with find command and list it down to the subdirectories.
SRCDIRS  := $(shell find . -type d)
#List all cpp files with the foreach command based on the SRCDIRS
CPPS   = $(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.cpp))
#Remove directories not to make.
SRCS = $(filter-out $(NOMAKEDIR), $(CPPS))
#Get the name of the directory containing the SRCS.
DIRS = $(dir $(SRCS))
#Create a directory for binary with the same structure as the cpp files.
BINDIRS = $(addprefix $(OBJDIR)/, $(DIRS))
# Determine the object file name based on the SRCS
OBJS = $(addprefix $(OBJDIR)/, $(patsubst %.cpp, %.o, $(SRCS))) # patsubstは空白で区切られたものか指定の文字列に置換
#Run through the object directory with find command and list it down to the subdirectories.
ICARATDIRS  := $(shell find $(ICARATPATH) -type d)
#List all cpp files with the foreach command based on the ICARATDIRS
ICARATOBJS   = $(foreach dir2, $(ICARATDIRS), $(wildcard $(dir2)/*.o))
# Make the dependency file .d from the .o file
DEPS = $(OBJS:.o=.d)

default:
	@[ -d  $(OBJDIR)   ] || mkdir -p $(OBJDIR)
	@[ -d "$(BINDIRS)" ] || mkdir -p $(BINDIRS)
	@make all --no-print-directory
	@echo "make done!!"

all : $(OBJS) $(TARGET)

$(TARGET): $(OBJS) $(ICARATOBJS)
	$(GCC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	$(GCC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

clean:
	@rm -rf $(TARGET) $(TILS) $(OBJDIR)

-include $(DEPS)