//面向对象

#include <iostream>

using namespace std;

class Box
{
    public:
        double length;   // 长度
        double breadth;  // 宽度
        double height;   // 高度
        // 成员函数声明，如果定义在类外，必须声明
        double get(void);
        void set(double len, double bre, double hei);
        double getVolume(void)
        {
            return length * breadth * height;
        }
};
double Box::get(void)
{
    return length * breadth * height;
}
void Box::set(double len, double bre, double hei)
{
    length = len;
    breadth = bre;
    height = hei;
}

void runoob_classes_objects()
{
    Box Box1;
    Box Box2;
    double volume = 0.0;     // 用于存储体积

    // box 1 详述
    Box1.height = 5.0; 
    Box1.length = 6.0; 
    Box1.breadth = 7.0;

    volume = Box1.height * Box1.length * Box1.breadth;
    cout << "Box1 的体积：" << volume <<endl;

    Box2.set(16.0, 8.0, 12.0);
    volume = Box2.get();
    cout << "Box2 的体积：" << volume << endl;
}

class BaseBox {
    public:
        //公有成员，公有成员在程序中任何地方都能访问，无需通过成员函数读写

    protected:
        // 受保护成员，区别体现在继承
        // 如果没有继承，它和 private 一样（外人不可见）。
        // 如果有继承，子类（派生类）可以访问父类的 protected 成员，但不能访问 private 成员。
        double width;

    private:
        // 私有成员，私有成员对类外完全封闭，外部代码无法读取、修改或调用，派生类同样无权直接访问。
        // 只有类自身的成员函数与被授予友元权限的实体能够操作这些内容。类中若未使用任何访问说明符，成员默认为私有
        // 实际操作中，我们一般会在私有区域定义数据，在公有区域定义相关的函数，以便在类的外部也可以调用这些函数

};

class SmallBox:BaseBox // SmallBox 是 BaseBox 的派生类
{
   public:
        double getSmallWidth(void)
        {
            return width;
        }
        void setSmallWidth(double wid)
        {
            width = wid;
        }
};

class Parent {
public:
    int pub_var;
protected:
    int pro_var;
private:
    int pri_var; // 只有 Parent 自己能动

public:
    Parent() { pub_var = 1; pro_var = 2; pri_var = 3; }
};
// 1. 公有继承（Public Inheritance）- 最常见，原汁原味
class ChildA : public Parent {
public:
    void test() {
        cout << pub_var << endl; // OK
        cout << pro_var << endl; // OK
        // cout << pri_var << endl; // 错误！父类私有成员不可见
    }
};

// 2. 受保护继承（Protected Inheritance）- 大家都变成了受保护
class ChildB : protected Parent {
public:
    void test() {
        cout << pub_var << endl; // OK，但在 ChildB 看来，它是 protected
        cout << pro_var << endl; // OK
    }
};

// 3. 私有继承（Private Inheritance）- 大家都变成了私有
class ChildC : private Parent {
public:
    void test() {
        cout << pub_var << endl; // OK，但在 ChildC 看来，它是 private
        cout << pro_var << endl; // OK，但在 ChildC 看来，它是 private
    }
};

void runoob_class_access_modifiers()
{
    ChildA a;
    cout << a.pub_var << endl; // OK，外部可以访问
    // cout << a.pro_var << endl; // 错误，外部不可访问 protected

    ChildB b;
    // cout << b.pub_var << endl; // 错误！因为是 protected 继承，pub_var 在外部变成了 protected

    ChildC c;
    // cout << c.pub_var << endl; // 错误！因为是 private 继承，pub_var 在外部变成了 private

}

class Rectangle
{
public:
    float getLength()
    {
        return length;
    }
    void setLength(float l)
    {
        length = l;
    }
    friend void printWidth( Rectangle r );
    void printInfo()
    {
        cout << "Object is being created, length = " << length << ", width = " << width << endl;
    }
    Rectangle()//无输入的构造函数
    {
        printInfo();
    }
    // Rectangle(float l)
    // {
    //     length = l;
    //     printInfo();
    // }
    Rectangle(float l): length(l)//使用初始化列表来初始化字段
    {
        printInfo();
    }
    Rectangle(float l, float w): length(l), width(w)//使用初始化列表来初始化字段
    {
        printInfo();
    }
    ~Rectangle(void)//析构函数，它会在每次删除所创建的对象时执行，它不会返回任何值，也不能带有任何参数
    {
        cout << "Object is being deleted" << endl;
    }
private:
    float length;
    float width;
};

void runoob_cpp_constructor_destructor()//类的构造
{
    Rectangle rectangle1;
    Rectangle rectangle2(10);
    Rectangle rectangle3(10, 0.1);
}
class Line
{
public:
    float getLength()
    {
        return *ptr;
    }
    void setLength(float l)
    {
        *ptr = l;
    }
    void printInfo()
    {
        cout << "Object is being created, length = " << *ptr << endl;
    }
    Line(void)
    {
        ptr = new float;
        *ptr = 0;
        printInfo();
    }
    Line(float l)
    {
        ptr = new float;
        *ptr = l;
        printInfo();
    }
    Line(const Line &obj)//拷贝构造函数
    {
        ptr = new float;
        *ptr = *obj.ptr; // 拷贝值
        printInfo();
    }
    ~Line(void)//析构函数，它会在每次删除所创建的对象时执行，它不会返回任何值，也不能带有任何参数
    {
        cout << "Object is being deleted" << endl;
    }
private:
    float *ptr;
};
void runoob_cpp_copy_constructor()//拷贝构造函数
{
    //如果在类中没有定义拷贝构造函数，编译器会自行定义一个。如果类带有指针变量，并有动态内存分配，则它必须有一个拷贝构造函数。
    // Line line1();//这其实是函数声明
    Line line1(10);//这才是Line的默认构造
    line1.getLength();
    Line line2 = line1;
}

void printWidth( Rectangle r )
{
    cout << "Width of Rectangle : " << r.width <<endl;
}

void runoob_cpp_friend_functions()//友元函数
{
    //类的友元函数是定义在类外部，但有权访问类的所有私有（private）成员和保护（protected）成员。
    //尽管友元函数的原型有在类的定义中出现过，但是友元函数并不是成员函数。
    //友元可以是一个函数，该函数被称为友元函数；友元也可以是一个类，该类被称为友元类，在这种情况下，整个类及其所有成员都是友元。
    //如果要声明函数为一个类的友元，需要在类定义中该函数原型前使用关键字 friend
    Rectangle rectangle(10, 20);
    printWidth(rectangle);
    //friend class ClassTwo;
    //声明类 ClassTwo 的所有成员函数作为类 ClassOne 的友元，需要在类 ClassOne 的定义中放置如下声明
}
inline int Max(int x, int y)
{
   return (x > y)? x : y;
}
void runoob_cpp_inline_functions()
{
    //如果已定义的函数多于一行，编译器会忽略 inline 限定符。
   cout << "Max (20,10): " << Max(20,10) << endl;
   cout << "Max (0,200): " << Max(0,200) << endl;
   cout << "Max (100,1010): " << Max(100,1010) << endl;
}

class Box_
{
   public:
      // 构造函数定义
      Box_(double l=2.0, double b=2.0, double h=2.0)
      {
         cout <<"调用构造函数。" << endl;
         length = l;
         breadth = b;
         height = h;
      }
      double Volume()
      {
         return length * breadth * height;
      }
      bool compare(Box_ box)
      {
         return this->Volume() > box.Volume();
      }
   private:
      double length;     // 宽度
      double breadth;    // 长度
      double height;     // 高度
};


void runoob_cpp_this_pointer()
{
    //在 C++ 中，this 指针是一个特殊的指针，它指向当前对象的实例。
    //当一个对象的成员函数被调用时，编译器会隐式地传递该对象的地址作为 this 指针。
    //友元函数没有 this 指针，因为友元不是类的成员，只有成员函数才有 this 指针。
    Box_ Box1(3.3, 1.2, 1.5);    // 声明 box1
    Box_ Box2(8.5, 6.0, 2.0);    // 声明 box2

    if(Box1.compare(Box2))
    {
        cout << "Box2 的体积比 Box1 小" <<endl;
    }
    else
    {
        cout << "Box2 的体积大于或等于 Box1" <<endl;
    }
}

class MyClass {
public:
    int data;

    void display() {
        std::cout << "Data: " << data << std::endl;
    }
};

void runoob_cpp_pointer_to_class()
{
    MyClass *ptr = new MyClass;//动态分配内存创建类对象
    ptr->data = 42;
    ptr->display();
    delete ptr;//释放动态分配的内存
}

void runoob_cpp_static_members()
{
    //我们可以使用 static 关键字来把类成员定义为静态的。当我们声明类的成员为静态时，这意味着无论创建多少个类的对象，静态成员都只有一个副本。
    //如果把函数成员声明为静态的，就可以把函数与类的任何特定对象独立开来。
    //静态成员函数即使在类对象不存在的情况下也能被调用，静态函数只要使用类名加范围解析运算符 :: 就可以访问。
    //静态成员函数没有 this 指针，只能访问静态成员（包括静态成员变量和静态成员函数）。
}

int main()
{
    // https://www.runoob.com/cplusplus/cpp-tutorial.html
    // runoob_classes_objects();//https://www.runoob.com/cplusplus/cpp-classes-objects.html
        // runoob_class_access_modifiers();//https://www.runoob.com/cplusplus/cpp-class-access-modifiers.html
        // runoob_cpp_constructor_destructor();//https://www.runoob.com/cplusplus/cpp-constructor-destructor.html
        // runoob_cpp_copy_constructor();//https://www.runoob.com/cplusplus/cpp-copy-constructor.html
        // runoob_cpp_friend_functions();//https://www.runoob.com/cplusplus/cpp-friend-functions.html
        // runoob_cpp_inline_functions();//https://www.runoob.com/cplusplus/cpp-inline-functions.html
        // runoob_cpp_this_pointer();//https://www.runoob.com/cplusplus/cpp-this-pointer.html
        // runoob_cpp_pointer_to_class();//https://www.runoob.com/cplusplus/cpp-pointer-to-class.html
        runoob_cpp_static_members();//https://www.runoob.com/cplusplus/cpp-static-members.html
    //https://www.runoob.com/cplusplus/cpp-inheritance.html
    return 0;
}