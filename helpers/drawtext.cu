#include <stdio.h>
#include <GL/glut.h>

#define DIM 1024


void drawText(float x, float y, char *string)
{
    char *c;
    glRasterPos2f(x, y);
    for(c=string;*c!='\0';c++)
    {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
    }
}

void drawTextCentered(char *s)
{
    
    drawText((-15.0*(sizeof(s)-1)/2.0)/DIM, 0, s);
}


void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    char st[] = "Loading CPU fractal..";
    //printf("size of s = %lu\n", sizeof(s)-1);
    //drawText((-15.0*(sizeof(s)-1)/2.0)/DIM, 0, s);
    
    drawTextCentered(st);
    glFlush();

}

int main(int argc, char** argv)
{
    // Initialize OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowSize(DIM, DIM);


    // Create first window
    glutCreateWindow("GPU | Time to render:\t---");
    glutDisplayFunc(display);

    glClearColor(0.1, 0.1, 0.1, 0.1);
    glutMainLoop();
    return(0);
}