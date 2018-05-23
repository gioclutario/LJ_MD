import numpy as np
import math
import random
# import pybinding as pb
# import matplotlib.pyplot as plt
import sys

""" A Lennard Jones Simulation in 2D """
class LJ_2D_Sim_Numpy():
    def __init__(self,sigma,eps,temperature,duration,dt,ip,iv,ncells,adist,rfreq,efreq,rc):
        self.sigma = sigma
        self.adist = adist/self.sigma
        self.eps = eps
        self.temperature = temperature
        self.duration = duration
        self.dt = dt
        self.ip = ip
        self.iv = iv
        self.ncells = ncells
        #Box size = ncells * adist
        self.boxsize = ncells * self.adist
        self.natoms = 2*(ncells**2)
        self.iterations = self.duration/self.dt
        self.rfreq = rfreq
        self.efreq = efreq
        self.rc = rc
        #LJ potential
        self.urc = 4.0 * (1.0/(self.rc**12) - 1.0/(self.rc**6))
        #derivative of LJ potential
        self.dudrc = (-48/self.rc) * (1.0/(self.rc**12) - 0.5/(self.rc**6))

    """ Position initialization function """
    def initial_position(self):
        #If condition to initialize lattice if ip is 1
        if self.ip == 1:
            #First of 2 initial atoms to build lattice with
            atom_1 = [0.0,0.0]
            #Second initial atom
            atom_2 = [self.adist/2,self.adist/2]
            #Empty list to contain lattice coordinates
            latticeList = []
            #Nested for loops to build lattice
            for atomx in range(self.ncells):
                for atomy in range(self.ncells):
                    #Formula to create new atom coordinates with initial position @ atom_1
                    newAtom_1 = [atom_1[0] + (self.adist*(atomx)),atom_1[1] + (self.adist*(atomy))]
                    #Formula to create new atom coordinates with initial position @ atom_2
                    newAtom_2 = [atom_2[0] + (self.adist*(atomx)),atom_2[1] + (self.adist*(atomy))]
                    #Appends the new coordinates to the empty list
                    latticeList.append(newAtom_1)
                    latticeList.append(newAtom_2)
            #converts list of coordinates into an array of similar dimensions
            latticeArray = np.array(latticeList)

            return latticeArray
        #If condition to read gro file of lattice coords if ip is 0
        # elif self.ip == 0:
        #     with open(sys.argv[1],"r") as gro_file1:
        #
        #     pass

    """ Momentum initialization function """
    def initial_momenta(self,ncells):
        # If fucntion to psuedo-randomly initialze momenta if iv = 1
        if self.iv == 1:
            u_0 = math.sqrt(2*self.temperature)
            #Empty list that will contain list of momenta for each partcile
            momentaList = []
            #Counter for the total momenta in the x-direction
            totalPX = 0.0 #Is this needed?
            #Counter for the total momenta in the y-direction
            totalPY = 0.0 #Is this needed?
            for atom in range(ncells):
                #Empty list that will contain the momenta of atom
                atomMomenta =[]
                #Generates random theta value
                theta = 2*math.pi*random.random()
                #Produces a random point within the unit circle
                rand_point = [math.cos(theta),math.sin(theta)]
                    # point_magnitude = (rand_point[0]**2) + (rand_point[1]**2)
                # sp = math.sqrt((1.0-point_magnitude))*2.0
                PX = rand_point[0] * u_0
                PY = rand_point[1] * u_0
                totalPX += PX
                totalPY += PY
                atomMomenta.append(PX)
                atomMomenta.append(PY)
                momentaList.append(atomMomenta)
            momentaArray = np.array(momentaList)
            return momentaArray
        # if self.iv == 0:
        #     return

    """ Force initialization function """
    def initial_forcesV2(self,lattice):
        #Intialization total potential energy
        ut = 0.0
        #Initialization of Force Array in X & Y direction
        iforcesArrayX = np.zeros((len(lattice),len(lattice)))
        iforcesArrayY = np.zeros((len(lattice),len(lattice)))
        #Main for loop to iterate over every atom
        for atom1 in range(len(lattice)):
            #Nested for loop to iterate over every potential pairwise interaction with atom1
            for atom2 in range(atom1+1,len(lattice)):
                #Calculates distance b/w atom1 and atom2 in x and y direction
                drx = lattice[atom1][0] - lattice[atom2][0]
                dry = lattice[atom1][1] - lattice[atom2][1]
                #Implements periodic boundary condition
                drx -= self.boxsize*round(drx/self.boxsize)
                dry -= self.boxsize*round(dry/self.boxsize)
                #Calculates the magnitude of the calculated distances
                r2 = (drx**2)+(dry**2)
                #If state to ensure that if r2 is within bounds of cutoff range, potential will be calculate based off of LJ potential equation otherwise value is 0.
                if r2 <= self.rc**2:
                    #Initializing components of LJ potential
                    r6r = (1.0)/(r2**3)
                    r12r = r6r ** 2
                    r = math.sqrt(r2)
                    #Calculating LJ potential
                    u = (r12r - r6r)*4.0 - self.urc - (r - self.rc) * self.dudrc
                    #Adding LJ potential to total PE
                    ut += u
                    #Calculating derivative of LJ potential
                    t = 48.0*(r12r - 0.5*r6r)/r2 + (self.dudrc)/r
                    #Initializing each array element with calculated force
                    iforcesArrayX[atom1][atom2] = drx*t
                    iforcesArrayY[atom1][atom2] = dry*t
                #Else statement to initialze array element with force value of 0 if outside of cut off range
                else:
                    iforcesArrayX[atom1][atom2] = 0
                    iforcesArrayY[atom1][atom2] = 0
                #Implements Newton's 3rd law
                iforcesArrayX[atom2][atom1] = -iforcesArrayX[atom1][atom2]
                iforcesArrayY[atom2][atom1] = -iforcesArrayY[atom1][atom2]
        return iforcesArrayX,iforcesArrayY

    """ Sub function to calculate the total force for a given array of forces """
    def totalForce(self,forceX,forceY):
        #Initializes total force Array containing the total forces of each atom in X & Y direction
        tforceArray = np.zeros([self.natoms,2])
        #Atom counter to iterate over each atom
        atomCount = 0
        #For loop to iterate over all atoms
        for atom1 in range(self.natoms):
            #total Force counter for X & Y
            tForceX = 0.0
            tForceY = 0.0
            #second for loop to iterate over all atoms for pair wise interactions
            for atom2 in range(self.natoms):
                #Sums X & Y forces of each pair wise interaction to totalForce
                tForceX += forceX[atom1][atom2]
                tForceY += forceY[atom1][atom2]
            #Equates each tForce to respective atom # in tforceArray
            tforceArray[atomCount][0] = tForceX
            tforceArray[atomCount][1] = tForceY
            #Increases atomcount by one to continue totalforce calculations
            atomCount += 1
        return tforceArray

    """Leap frog Algorithm which takes in position(lattice), force, and momenta of a system and outputs
    the resulting new force & momenta of the system after a given amount of time set by duration."""

    def leapFrogAlgo(self,lattice,forceX,forceY,momenta):
        #Kinetic energy counter
        ke = 0.0

        #Total acceleration in X-direction counter
        accelX = 0.0

        #Total acceleration in Y-direction counter
        accelY = 0.0

        #For loop to obtain total acceleration of i'th atom

        forceList = self.totalForce(forceX,forceY)

        #Main for loop which loops through self.duration in intervals of self.dt
        for t in np.arange(0,self.duration,self.dt):
            # print('')
            #Empty list which will store the x_(i+1) positions
            newPosition = []
            #Empty list which will store the v_(i+1) positions
            newVelocity = []

            for i in range(len(forceList)):
                accelX += forceList[i][0]
                accelY += forceList[i][1]

            #Total non-shifted X-acceleration
            daccelX = accelX/float(len(lattice))

            #Total non-shifted Y-acceleration
            daccelY = accelY/float(len(lattice))

            #Nested foor loop whcih will iterate through each atom within the initialized lattice
            for i in range(self.natoms):

                #Leap Frog Algorithm's method for calculating the next position
                newPos = [lattice[i][0] + momenta[i][0]*self.dt + (1/2)*(forceList[i][0]-daccelX)*(self.dt**2),
                lattice[i][1] + momenta[i][1]*self.dt + (1/2)*(forceList[i][1]-daccelY)*(self.dt**2)]
                # print("This is the value of newPos = {}".format(newPos))
                newPosition.append(newPos)
                # print("This is what newPosition looks like = {}".format(newPosition))
            #Updates lattice with new Position coordinates
            if newPosition[0][0] == lattice[0][0]:
                print('New position has no changed')
            lattice = newPosition

            #Apply a sub function that will append the results of the completed newPos into a text file
            newForceX,newForceY = self.initial_forcesV2(lattice)
            ntForce = self.totalForce(newForceX,newForceY)

            for i in range(self.natoms):
                newVel = [momenta[i][0] + (1/2)*(ntForce[i][0]+forceList[i][0]-daccelX)*self.dt,
                momenta[i][1] + (1/2)*(ntForce[i][1]+forceList[i][0]-daccelY)*self.dt]
                # print("This is the value of newVel = {}".format(newVel))
                newVelocity.append(newVel)
                # print('This is what newVelocity looks like = {}'.format(newVelocity))
            with open(sys.argv[2],"a") as positionTrajectory:
                positionTrajectory.write('MD sim of {} Argon atoms, t = {}\n'.format(self.natoms,t))
                positionTrajectory.write('{}\n'.format(self.natoms))
                for i in range(len(newPosition)):
                    trajOutput ='{resNum:>5d}{resName:>5s}{atomName:>5s}{atomNum:>5d}{posX:8.3f}{posY:8.3f}{posZ:8.3f}\n'.format(resNum = i+1,resName = 'ARGON',atomName ='Ar', atomNum = i+1 ,posX = newPosition[i][0] ,posY =newPosition[i][1], posZ = 1)
                    positionTrajectory.write(trajOutput)
                positionTrajectory.write('{0} {0} {0}\n'.format(self.boxsize))
            #Updates momenta with new velocities

            momenta = newVelocity

            #Updates forceList(a_i) with ntForce(a_i+1) for next iteration
            forceList = ntForce
        #Apply a sub funciton that will append the rsults of the completed newVel into a text file
        with open(sys.argv[1],"a") as positionOutput:
            positionOutput.write('MD Sim {} Argons\n'.format(self.natoms))
            for i in range(len(newPosition)):
                posOutput ='{resNum:>5d}{resName:>5s}{atomName:>5s}{atomNum:>5d}{posX:8.3f}{posY:8.3f}{velX:8.4f}{velY:8.4f}{velZ:8.4f}\n'.format(resNum = i+1,resName = 'ARGON',atomName ='Ar', atomNum = i+1 ,posX = newPosition[i][0] ,posY =newPosition[i][1], posZ = 1,velX =momenta[i][0],velY = momenta[i][1],velZ = 0)
                positionOutput.write(posOutput)
            positionOutput.write('{0} {0} {0}\n'.format(self.boxsize))
        return ke,accelX,accelY

    """Plotter function to visualize any given 2D lattice"""
    def Plotter(self,lattice):
        x = lattice[:,0]
        y = lattice[:,1]
        plt.scatter(x,y)
        latplot = plt.show()
        return latplot

def main():
    #(self,sigma,eps,temperature,duration,dt,ip,iv,ncells,adist,rfreq,efreq,rc)
    instance = LJ_2D_Sim_Numpy(0.34,120.0,0.5,1.0,0.001,1,1,5,0.53,100,10,1.6/1.414)
    lattice = instance.initial_position()
    # with open('lattice.txt','a') as latticeFile:
    #     latticeFile.write('{}'.format(lattice))
    imomenta = instance.initial_momenta(len(lattice))
    # with open('velocity.txt','a') as velocityFile:
    #     velocityFile.write('{}'.format(imomenta))
    iforceX,iforceY = instance.initial_forcesV2(lattice)
    sim = instance.leapFrogAlgo(lattice,iforceX,iforceY,imomenta)

if __name__ == '__main__':
    main()
