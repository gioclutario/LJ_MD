import numpy as np
import math
import random
import os
import sys
import re

""" A Lennard Jones Simulation in 2D """
class LJ_2D_Sim_Numpy():
    vscaler = 1
    def __init__(self,sigma,eps,temperature,duration,dt,ip,iv,ncells,numDensity,rfreq,efreq,rc):
        #Stores initialied sigma value
        self.sigma = sigma
        #Stores initialized episolon value
        self.eps = eps
        #Stores Target temperature of the system
        self.temperature = temperature
        #Duration of simulation
        self.duration = duration
        #Time step of simulation
        self.dt = dt
        #Flag that will determine if initial strucutre needs to be produced or read from input File
        self.ip = ip
        #Flag that will determine if iniital velocity neeeds to be produced or read from input File
        self.iv = iv
        #Stores target atom count of PBC box
        self.ncells = ncells
        #Box size = ncells * adist
        #Calculates total num of atoms in system
        self.natoms = 2*(ncells**2)
        #Calculates total number of iterations for simulation
        self.iterations = self.duration/self.dt
        #Stores the Density of system
        self.numDensity = numDensity
        #Stores the frequnecy of energy output
        self.efreq = efreq
        #Stores the frequency of position output
        self.rfreq = rfreq
        #Stores cutoff range value
        self.rc = rc*self.sigma
        #LJ potential
        self.urc = 4.0 * (1.0/(self.rc**12) - 1.0/(self.rc**6))
        #derivative of LJ potential
        self.dudrc = (-48/self.rc) * (1.0/(self.rc**12) - 0.5/(self.rc**6))
    """ Position initialization function """
    def initial_latticeV2(self):
        if self.ip == 1:
            #Generates appropriate a_dist based off of reduced density
            a_dist = math.sqrt(2/self.numDensity)
            #Intial position of first atom
            atom_1 = [0.0,0.0]
            #initial position of second ato
            atom_2 = [a_dist/2,a_dist/2]
            #Initializes list that will eventually store positions of all atoms within system
            latticeList = []
            #Main for loop that iterates through all atoms in system
            for atomx in range(self.ncells):
                #Nested for loop that iterates through all atoms in system
                for atomy in range(self.ncells):
                    #Calculates initial positions of all atoms based off of first and second atom initial positions
                    newAtom_1 = [atom_1[0]+(a_dist*(atomx)),atom_1[1]+(a_dist*(atomy))]
                    newAtom_2 = [atom_2[0]+(a_dist*(atomx)),atom_2[1]+(a_dist*(atomy))]
                    #Appends calculated initial positions into latticeList
                    latticeList.append(newAtom_1)
                    latticeList.append(newAtom_2)
            #converts latticeList into an array
            latticeArray = np.array(latticeList)
            #returns latticeArray & calculated boxsize
            return latticeArray,a_dist*self.ncells
        elif self.ip == 0:
            #Empty list that will eventually contain all of the parsed coordinates
            latticeList = []
            #Context manager for input file
            with open('final.gro',"r") as latInput:
                #Parses through each individual line of the file
                lines = latInput.readlines()

                #A for loop to iterate through each line in lines with a counter via the enumerate function
                for count,item in enumerate(lines):
                    #If statement to ensure tha header & box size lines are not considered
                    if count !=0 and count != len(lines)-1:
                        #coords variable to store all of the posX & posY columns
                        coords = [float(item[21:28].strip()),float(item[29:36].strip())]
                        #Appends coords to latticeList
                        latticeList.append(coords)
                    if count ==len(lines)-1:
                        boxsize = float(item.split()[1])
            #Converts lattice list into an array
            latticeArray = np.array(latticeList)
            return latticeArray,boxsize/self.sigma
    """ Momentum initialization function """
    def initial_momenta(self,ncells):
        # If fucntion to randomly initialze momenta if iv = 1
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
        #if condition to read in initial momenta file
        if self.iv == 0:
            #Empty momentaList which will contain the initial momenta of each atom
            momentaList = []
            #Context manager for input file
            with open('final.gro',"r") as velInput:
                #Reads through each line of the file
                lines = velInput.readlines()
                #For loop to iterate through each line in lines with a counter
                for count,item in enumerate(lines):
                    #If statement to ensure that header and box are not considered
                    if count !=0 and count != len(lines)-1:
                        coords = [float(item[45:52].strip()),float(item[53:60].strip())]
                        momentaList.append(coords)
            return np.array(momentaList)

    """ Force initialization function """
    def initial_forcesV2(self,lattice,boxsize):
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
                drx -= boxsize*round(drx/boxsize)
                dry -= boxsize*round(dry/boxsize)
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
        return iforcesArrayX,iforcesArrayY,ut

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
    def leapFrogAlgo(self,lattice,forceX,forceY,momenta,boxsize):
        #Total acceleration in X-direction counter
        accelX = 0.0
        #Total acceleration in Y-direction counter
        accelY = 0.0
        forceList = self.totalForce(forceX,forceY)
        rfreqCounter = 0
        efreqCounter = 0
        #Main for loop which loops through self.duration in intervals of self.dt
        for t in np.arange(0,self.duration,self.dt):
            #Empty list which will store the x_(i+1) positions
            newPosition = []
            #Empty list which will store the v_(i+1) positions
            newVelocity = []
            #Counter for kinetic energy that will reset with each time step
            kineticEnergy = 0.0
            #Counter for potential energy that will reset with each time step
            potentialEnergy = 0.0
            #For loop to sum up the total acceleration of the system
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
                newPos = [lattice[i][0] + momenta[i][0]*self.dt + (1/2)*(forceList[i][0]-daccelX)*(self.dt**2),lattice[i][1] + momenta[i][1]*self.dt + (1/2)*(forceList[i][1]-daccelY)*(self.dt**2)]
                #Appends newPos to newPosition list
                newPosition.append(newPos)
                #Calculates the kinetic energy of i'th atom and adds to KE counter
                kineticEnergy += 0.5*((momenta[i][0]**2)+(momenta[i][1]**2))
            #Updates lattice with new Position coordinates
            lattice = newPosition
            #Apply a sub function that will append the results of the completed newPos into a text file
            newForceX,newForceY,ut = self.initial_forcesV2(lattice,boxsize)
            potentialEnergy += ut
            ntForce = self.totalForce(newForceX,newForceY)
            if LJ_2D_Sim_Numpy.vscaler == 1:
                iTemp = (kineticEnergy/self.natoms)
                vscaled = math.sqrt(self.temperature/iTemp)
                for i in range(self.natoms):
                    #Leap Frog Algorithm's method for calculating the next velocity
                    newVel = [(momenta[i][0] + (1/2)*(ntForce[i][0]+forceList[i][0]-daccelX)*self.dt)*vscaled,(momenta[i][1] + (1/2)*(ntForce[i][1]+forceList[i][1]-daccelY)*self.dt)*vscaled]
                    #Appends newVel to the append to the newVelocity list
                    newVelocity.append(newVel)
            else:
                for i in range(self.natoms):
                    #Leap Frog Algorithm's method for calculating the next velocity
                    newVel = [momenta[i][0] + (1/2)*(ntForce[i][0]+forceList[i][0]-daccelX)*self.dt, momenta[i][1] + (1/2)*(ntForce[i][1]+forceList[i][1]-daccelY)*self.dt]
                    #Appends newVel to the append to the newVelocity list
                    newVelocity.append(newVel)
            #If statemetn to determine whether or not to write the file based off of rfreq
            if rfreqCounter%self.rfreq == 0:
                with open('traj.gro',"a") as positionTrajectory:
                    positionTrajectory.write('MD sim of {} Argon atoms, t = {}\n'.format(self.natoms,t))
                    positionTrajectory.write('{}\n'.format(self.natoms))
                    for i in range(len(newPosition)):
                        trajOutput ='{resNum:>5d}{resName:>5s}{atomName:>5s}{atomNum:>5d}{posX:8.3f}{posY:8.3f}{posZ:8.3f}\n'.format(resNum = i+1,resName = 'ARGON',atomName ='Ar', atomNum = i+1 ,posX = newPosition[i][0]*self.sigma ,posY =newPosition[i][1]*self.sigma, posZ = 1)
                        positionTrajectory.write(trajOutput)
                    positionTrajectory.write('{0} {0} {0}\n'.format(boxsize*self.sigma))
            if efreqCounter%self.efreq == 0:
                with open('te.txt',"a") as teFile,open('pe.txt',"a") as peFile,open('ke.txt',"a") as keFile,open('temp.txt','a') as tempFile:
                    peFile.write('{time:>3f} {pe:>8.3f}\n'.format(time = t,pe = potentialEnergy))
                    keFile.write('{time:>3f} {ke:>8.3f}\n'.format(time = t,ke = kineticEnergy))
                    teFile.write('{time:>3f} {te:8.3f}\n'.format(time = t,te = kineticEnergy + potentialEnergy))
                    tempFile.write('{time:>3f} {temp:8.3f}\n'.format(time = t, temp = (kineticEnergy)/self.natoms))
            #Updates momenta with new velocities
            momenta = newVelocity
            #Updates forceList(a_i) with ntForce(a_i+1) for next iteration
            forceList = ntForce
            rfreqCounter += 1
            efreqCounter += 1
        #Apply a sub funciton that will append the rsults of the completed newVel into a text file
        with open('final.gro',"a") as positionOutput:
            #Writes the header of the final output file
            positionOutput.write('MD Sim {} Argons\n'.format(self.natoms))
            #for loop to iterate through every single newPosition
            for i in range(len(newPosition)):
                #Writes the properly formatted info of the atom and its position&velocity
                posOutput ='{resNum:>5d}{resName:>5s}{atomName:>5s}{atomNum:>5d}{posX:8.3f}{posY:8.3f}{posZ:8.3f}{velX:8.4f}{velY:8.4f}{velZ:8.4f}\n'.format(resNum = i+1,resName = 'ARGON',atomName ='Ar', atomNum = i+1 ,posX = newPosition[i][0] ,posY =newPosition[i][1], posZ = 1,velX =momenta[i][0],velY = momenta[i][1],velZ = 0)
                positionOutput.write(posOutput)
            #Writes the boxsize ender for the final output file
            positionOutput.write('{0} {0} {0}\n'.format(boxsize))
        #Implement output reader that
        return

    """Method for reading in an energy file to analyze the fluctation and drift of the energy within system"""
    def energyReader(self,file):
        #Context manager for inputFile
        with open(file,'r') as inputFile:
            #Initializing counters for sum of Energy, frame count, rmsf sum energy, and initial energy at t=0
            sumE = 0.0
            frames = 0
            rmsfsumE = 0.0
            firstLine = 0
            #For loop to iterate through each line of the file
            for count,line in enumerate(inputFile):
                #If condition to store the value of E at t=0
                if count == 0:
                    #Stores value of E
                    firstLine += float(line[10:17].strip())
                #Adds up the energy of each line within the file
                sumE += float(line[10:17].strip())
                #Adds to the frame counter to tally up number of frames
                frames += 1
            #Saves lastLine of file
            lastLine = float(line[10:17].strip())
            #Calculates overall average of the system
            averageE = sumE/frames
            #Resets pointer to the beginning of the file
            inputFile.seek(0)
            #For loop to claculate rmsf and adds it to the rmsf counter
            for count,line in enumerate(inputFile):
                rmsfsumE += (float(line[10:17].strip())-averageE)**2
            #calculates the fluctation of the energy in the system
            fluctuationE = math.sqrt(rmsfsumE/frames)
            #Calculates the drift based off of the first and last line
            drift = lastLine - firstLine
        return fluctuationE/averageE,drift/averageE

    def vScaleOn():
        self.vscaler = 1

    def vScaleOff():
        self.vscaler = 0

    def readMode():
        self.iv = 0
        self.ip = 0

    def shortProdRun():
        self.vScaleOff()
        self.readMode()

    def productionRun():
        self.vScaleOff()
        self.rfreq = 50
        self.efreq = 50

def main():
    instanceArgs = []
    with open(sys.argv[1],'r') as inputFile:
        for count,line in enumerate(inputFile):
            arg = line.rstrip('\n').rsplit('=')[1]
            arg = re.sub(r"[\n\s]*","",arg)
            instanceArgs.append(arg)
    #(sigma,eps,temperature,duration,dt,ip,iv,ncells,numDensity,rfreq,efreq,rc,keFN,peFN,teFN,trajFN,finalFN)
    instance = LJ_2D_Sim_Numpy(float(instanceArgs[2]),float(instanceArgs[3]),float(instanceArgs[4]),float(instanceArgs[6]),float(instanceArgs[5]),int(instanceArgs[7]),int(instanceArgs[8]),int(instanceArgs[9]),int(instanceArgs[10]),int(instanceArgs[11]),float(instanceArgs[12]),float(instanceArgs[13]),instanceArgs[14],instanceArgs[15],instanceArgs[16],instanceArgs[17],)
    #Flag to determine if readMode has been initiated or not
    readMode = False
    infLoop = False
    counter = 0
    while True:
        lattice,bxsz = instance.initial_latticeV2()
        imomenta = instance.initial_momenta(len(lattice))
        iforceX,iforceY,ut = instance.initial_forcesV2(lattice,bxsz)
        sim = instance.leapFrogAlgo(lattice,iforceX,iforceY,imomenta,bxsz)
        fluctKE,driftKE = instance.energyReader('ke.txt')
        fluctPE,driftPE = instance.energyReader('pe.txt')
        fluctTE,driftTE = instance.energyReader('te.txt')
        with open('drift.txt','a') as driftFile,open('fluct.txt','a') as fluctFile:
            driftFile.write('{DriftK:>8.4f}{DriftP:>8.4f}{DriftT:>8.4f}'.format(DriftK=driftKE,DriftP=driftPE,DriftT=driftTE))
            fluctFile.write('{fluctK:>8.4f}{fluctP:>8.4f}{fluctT:>8.4f}'.format(fluctK=fluctKE,fluctP=fluctPE,fluctT=fluctTE))
        if fluctKE >= .1:
            instance.readMode()
            readMode = True
            counter += 1
        elif fluctKE <= .01:
            break
        elif counter > 50:
            break
if __name__ == '__main__':
    main()
