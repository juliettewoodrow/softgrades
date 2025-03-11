import React from 'react';
import { Card, Form } from 'react-bootstrap';
import { useState } from 'react';
import { on } from 'events';
import { object } from 'prop-types';

const ZeroToOneList = ({ length, stepSize }) => {
  return (
    <>
    {Array.from({ length: length }, (_, i) => {
      const value = (i * stepSize).toFixed(2);
      return <option key={i} value={value}>{value}</option>;
    })}
    </>
  )
}

const AssignmentCard = ({ assignmentType, assignmentNumber, score, onUpdateScore, mean, onUpdateMean, standardDeviation, onUpdateStandardDeviation, weight, onUpdateWeight }) => {
  const borderStyle = assignmentType === 'completed' ? 'success' : 'warning';
  const [shownScore, setShownScore] = useState(score);

    const handleWeightChange = (event) => {
      const val = event.target.value;
      onUpdateWeight(val);
    }
  
    const handleMeanChange = (event) => {
      onUpdateMean(event.target.value);
    };
  
    const handleStandardDeviationChange = (event) => {
      onUpdateStandardDeviation(event.target.value);
    };

    const handleScoreSubmit = (event) => {
      if (event.key !== 'Enter') return;
      const val = event.target.value;
      onUpdateScore(val);
    }

  
    const handleScoreChange = (event) => {
      const val = event.target.value;
      setShownScore(val);
    };

  // console.log("Assignment Card Re-rendering")

  return (
    <Card
      className="m-2 assignment-card"
      border={borderStyle}
    >
      <Card.Header>
        Assignment {assignmentNumber}
      </Card.Header>
      <Card.Body>
        <Form >
          <Form.Group>
            <Form.Label>Assignment Mean:</Form.Label>
            <Form.Control 
              as="select"
              value={mean}
              onChange={handleMeanChange}
            >
              <ZeroToOneList length={11} stepSize={0.10}/>
            </Form.Control>
          </Form.Group>
          <Form.Group>
            <Form.Label>Standard Deviation:</Form.Label>
            <Form.Control 
              as="select"
              value={standardDeviation}
              onChange={handleStandardDeviationChange}
            >
              {Array.from({ length: 13 }, (_, i) => 
                <option key={i} value={(i * 0.25) + 1}>{(i* 0.25) + 1}</option>
              )}
            </Form.Control>
          </Form.Group>
          {/* <Form.Group>
            <Form.Label>Assignment Weight:</Form.Label>
            <Form.Control 
              as="select"
              value={weight}
              onChange={handleWeightChange}
            >
              <ZeroToOneList length={11} stepSize={0.1}/>
            </Form.Control>
          </Form.Group> */}
          {assignmentType === 'completed' && (
            <Form.Group>
            <Form.Label>Score:</Form.Label>
            <Form.Control
              type="text"
              placeholder="Score (0-1)"
              // pattern="0(\.\d+)?|1(\.0+)?"
              title="Score must be a number between 0 and 1."
              value={shownScore}
              onChange={handleScoreChange}
              onKeyDown={handleScoreSubmit}
            />
          </Form.Group>
          )}
        </Form>
      </Card.Body>
    </Card>
  );
};

export default AssignmentCard;