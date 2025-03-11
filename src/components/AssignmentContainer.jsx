import React, { useState, useEffect } from 'react';
import AssignmentList from './AssignmentList';
import NormalDistributionPlot from './Plot';
import { Button } from 'react-bootstrap';

const MAX_NUM_ASSIGNMENTS = 10;

/**
 * Helper function to call your Firebase function:
 *  - Takes the `observed` object as input
 *  - Sends a POST request to your deployed function
 *  - Returns the JSON result (soft grade distribution, etc.)
 */
async function callFirebaseSoftGradeFunction(observed) {
  const functionUrl = "https://gradedistribution-df2lx4el5a-uc.a.run.app";
  try {
    const response = await fetch(functionUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(observed),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    console.log("Firebase function response:", response);
    return await response.json();
  } catch (error) {
    console.error("Error calling Firebase function:", error);
    throw error;
  }
}

const AssignmentContainer = () => {
  const [totalAssignments, setTotalAssignments] = useState(0);
  const [completedAssignments, setCompletedAssignments] = useState(0);
  const [canComputeSoftBelief, setCanComputeSoftBelief] = useState(true);
  const [softBelief, setSoftBelief] = useState([]);
  const [isCreatingSoftBelief, setIsCreatingSoftBelief] = useState(false);

  const [assnIds, setAssnIds] = useState([]);
  const [assignments, setAssignments] = useState([]);

  useEffect(() => {
    // If nothing is selected, just clear
    if (totalAssignments === 0 || completedAssignments === 0) {
      setAssignments([]);
      setAssnIds([]);
      return;
    }
  
    setAssignments((prevAssignments) => {
      const newAssignments = [];
  
      for (let i = 0; i < totalAssignments; i++) {
        // check if there's an existing assignment with id === i
        const existing = prevAssignments.find(a => a.id === i);
  
        if (existing) {
          // preserve the user-edited fields like mean, std, score, etc.
          // only update 'type' if it changed (completed vs future)
          newAssignments.push({
            ...existing,
            type: i < completedAssignments ? 'completed' : 'future'
          });
        } else {
          // This assignment didn't exist before; create a new one
          newAssignments.push({
            id: i,
            type: i < completedAssignments ? 'completed' : 'future',
            assignmentNumber: i + 1,
            mean: 0.80,
            standardDeviation: 1,
            weight: 1 / totalAssignments,
            score: i < completedAssignments ? 0 : '',
          });
        }
      }
      return newAssignments;
    });
  
    setAssnIds(Array.from({ length: totalAssignments }, (_, i) => i));
  }, [totalAssignments, completedAssignments]);
  

  // Update an individual assignment's property
  // const updateAssignment = (index, field, value) => {
  //   setAssignments((prevAssignments) => {
  //     const newAssignments = [...prevAssignments];
  //     newAssignments[index][field] = value; 
  //     return newAssignments;
  //   });
  // };

  const updateAssignment = (index, field, value) => {
    setAssignments((prevAssignments) => {
      console.log("Prev Assignments", prevAssignments);
      return prevAssignments.map((assignment, i) =>
        i === index ? { ...assignment, [field]: value } : assignment
      )
    }
    );
  };
  

  const renderCompletedAssignments = () => {
    return (
      <>
        <h4>Completed Assignments</h4>
        <div style={{ display: 'flex', flexDirection: 'row', overflowX: 'auto', padding: '0.5rem' }}>
          <AssignmentList
            assnIds={assnIds}
            type="completed"
            allAssignments={assignments}
            updateAssignment={updateAssignment}
          />
        </div>
      </>
    );
  };

  const renderFutureAssignments = () => {
    return (
      <>
        <h4>Future Assignments</h4>
        <div style={{ display: 'flex', flexDirection: 'row', overflowX: 'auto', padding: '0.5rem' }}>
          <AssignmentList
            assnIds={assnIds}
            type="future"
            allAssignments={assignments}
            updateAssignment={updateAssignment}
          />
        </div>
      </>
    );
  };

  const RenderAllAssignments = () => {
    console.log("SOFT BELIEF", softBelief)
    return (
      <div style={{ display: 'flex', flexDirection: 'row', overflowX: 'auto', padding: '0.5rem' }}>
        <div>
          {renderCompletedAssignments()}
        </div>
        <div>
          {renderFutureAssignments()}
        </div>

        {/* If we have a distribution for "student1", show NormalDistributionPlot */}
        {softBelief? 
          <NormalDistributionPlot gradeDistribution={softBelief} />
          : null}
      </div>
    );
  };

  /**
   * Transform the current assignments array into the 'observed' object
   * that your Python backend function expects.
   */
  const transformAssignmentsToObserved = (assignments) => {
    const assignment_stats = {};
    assignments.forEach((assn, i) => {
      assignment_stats[i] = {
        mean: parseFloat(assn.mean),
        std: parseFloat(assn.standardDeviation),
      };
    });

    // The student's scores come from completed assignments
    const studentScores = assignments
      .filter(a => a.type === 'completed')
      .map(a => parseFloat(a.score || 0));

    // const students = {
    //   student1: { scores: studentScores },
    // };

    // const scores = studentScores

    return { assignment_stats, studentScores };
  };

  /**
   * Button click handler:
   *  1. Sets loading
   *  2. Transforms assignments -> observed
   *  3. Calls the Firebase function
   *  4. Sets the softBelief state with the result
   *  5. Clears loading
   */
  const buttonClickCreateSoftBelief = async (assignments) => {
    try {
      setIsCreatingSoftBelief(true);  // show loading state
      const observed = transformAssignmentsToObserved(assignments);

      // Call the Firebase function instead of local computeSoftBelief
      const responseData = await callFirebaseSoftGradeFunction(observed);

      const finalGradeDistribution = responseData["finalGradeDistribution"]

      // Update state
      setSoftBelief(finalGradeDistribution);
      console.log("Soft belief (Firebase) returned:", responseData);

    } catch (error) {
      console.error("Error in buttonClickCreateSoftBelief:", error);
      alert("An error occurred while calculating soft grade. Please try again.");
    } finally {
      setIsCreatingSoftBelief(false); // hide loading state
    }
  };

  const renderSelectButtons = () => {
    return (
      <>
        <div className="mb-3">
          <label htmlFor="totalAssignments">Total number of assignments:</label>
          <select
            className="select-number"
            id="totalAssignments"
            value={totalAssignments}
            onChange={(e) => setTotalAssignments(Number(e.target.value))}
          >
            {Array.from({ length: MAX_NUM_ASSIGNMENTS }, (_, i) => (
              <option key={i} value={i + 1}>{i + 1}</option>
            ))}
          </select>
        </div>

        <div className="mb-3">
          <label htmlFor="completedAssignments">Number of completed Assignments:</label>
          <select
            className="select-number"
            id="completedAssignments"
            value={completedAssignments}
            onChange={(e) => setCompletedAssignments(Number(e.target.value))}
          >
            {Array.from({ length: totalAssignments }, (_, i) => (
              <option key={i} value={i}>{i}</option>
            ))}
          </select>
        </div>
      </>
    );
  };

  // console.log("Assignment Container Re-rendering");

  return (
    <div>
      {renderSelectButtons()}

      {totalAssignments > 0 && completedAssignments > 0 && (
        <>
          <p>Welcome to Soft Grades! Enter the information for all of your completed assessments. For future assessments, put in your best guess for the mean and standard deviation. Then we will calculate your Soft Grade for you. You can experiment with different values and see how that will impact your Soft Grade. </p>
          <RenderAllAssignments />

          {canComputeSoftBelief ? (
            <>
            <Button
              variant="primary"
              onClick={() => buttonClickCreateSoftBelief(assignments)}
              disabled={isCreatingSoftBelief || !canComputeSoftBelief}
            >
              {isCreatingSoftBelief ? "Calculating..." : "Create Soft Grade"}
            </Button>
            <div>{isCreatingSoftBelief ? "It takes 2-3 minutes to compute your soft grade" : ""}</div>
            </>
          ) : (
            <Button disabled>Create Soft Grade</Button>
          )}
        </>
      )}
    </div>
  );
};

export default AssignmentContainer;
