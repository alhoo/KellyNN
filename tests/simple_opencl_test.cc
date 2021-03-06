#include "simple_opencl_class.h"
#include "gtest/gtest.h"
#define NS (16)
#define SS (NS*NS)


namespace{


class opencl_class_test : public ::testing::Test {
    protected:
    // You can remove any or all of the following functions if its body
      // is empty.

        opencl_class_test() {
            // You can do set-up work for each test here.
        }

        virtual ~opencl_class_test() {
        // You can do clean-up work that doesn't throw exceptions here.
        }

        // If the constructor and destructor are not enough for setting up
        // and cleaning up each test, you can define the following methods:

        virtual void SetUp() {
        // Code here will be called immediately after the constructor (right
        // before each test).
        }

        virtual void TearDown() {
            // Code here will be called immediately after each test (right
            // before the destructor).
        }

        simple_opencl_class soc;
        // Objects declared here can be used by all tests in the test case for Foo.
};

TEST_F(opencl_class_test, setStateTest) {
    EXPECT_EQ(1, soc.test());
}


}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
