#include <stdio.h>

void main()
{
    int a = 1;
    printf("Enter before switch\r\n");
    switch(a)
    {
        printf("Begin of switch\r\n");

        case 0://过拧紧距离，刹车
            goto end_of_work;
            printf("End of case 0\r\n");
            break;

        case 1://过目标扭矩，刹车
            goto end_of_work;
            printf("End of case 1\r\n");
            break;

        default:
            printf("End of default\r\n");
            break;

        end_of_work:
            printf("In work\r\n");
            break;

        printf("End of switch\r\n");
    }
    printf("Enter after switch\r\n");
}