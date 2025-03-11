import React from 'react';
import AssignmentCard from './AssignmentCard';


const AssignmentList = ({ assnIds, type, allAssignments, updateAssignment }) => {

  const assignments = allAssignments.filter(assignment => assignment.type === type);

  let toAdd = 0
  if (type === "future") {
    toAdd = allAssignments.length - assignments.length
  }

  // console.log("Assignment List Re-rendering", allAssignments)

  return (
  <div style={{ display: 'flex', flexDirection: 'row', overflowX: 'auto', padding: '0.5rem' }}>
    {
      assignments.map((assigment, i) => {
        return (
          <AssignmentCard 
            key={assnIds[i + toAdd]}
            assnId={assnIds[i + toAdd]}
            assignmentType={type}
            assignmentNumber={assigment.assignmentNumber}
            mean={assigment.mean}
            standardDeviation={assigment.standardDeviation}
            score={assigment.score}
            onUpdateMean={(value) => updateAssignment(i + toAdd, 'mean', value)}
            onUpdateStandardDeviation={(value) => updateAssignment(i + toAdd, 'standardDeviation', value)}
            onUpdateScore={(value) => updateAssignment(i + toAdd, 'score', value)}
            weight={assigment.weight}
            onUpdateWeight={(value) => updateAssignment(i + toAdd, 'weight', value)}
          />
        )
      })
    }
  </div>
  )

};

export default AssignmentList;