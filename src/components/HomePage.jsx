// // HomePage.js
// import React from 'react';
// import { useNavigate } from 'react-router-dom';
// import { Container, Row, Col, Button } from 'react-bootstrap';

// const HomePage = () => {
//   const navigate = useNavigate();

//   return (
//     <div
//       className="app-container"
//       style={{
//         minHeight: '100vh',
//         display: 'flex',
//         alignItems: 'center',
//         justifyContent: 'center',
//         backgroundColor: '#f8f9fa',
//       }}
//     >
//       <Container>
//         <Row>
//           <Col className="text-center">
//             <h1>Soft Grades Web Application for Teachers and Students</h1>
//             <div style={{ marginTop: '2rem' }}>
//               <Button
//                 variant="primary"
//                 style={{ marginRight: '1rem' }}
//                 onClick={() => navigate('/teacher')}
//               >
//                 I&apos;m a teacher
//               </Button>
//               <Button
//                 variant="secondary"
//                 onClick={() => navigate('/student')}
//               >
//                 I&apos;m a student
//               </Button>
//             </div>
//           </Col>
//         </Row>
//       </Container>
//     </div>
//   );
// };

// export default HomePage;

import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Row, Col, Button, OverlayTrigger, Tooltip } from 'react-bootstrap';

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <div
      className="app-container"
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        // backgroundColor: '#f8f9fa',
      }}
    >
      <Container>
        <Row>
          <Col className="text-center">
            <h1>Soft Grades Web Application for Teachers and Students</h1>
            <div style={{ marginTop: '2rem' }}>
              <Button
                variant="primary"
                style={{ marginRight: '1rem' }}
                onClick={() => navigate('/teacher')}
              >
                I'm a teacher
              </Button>
              <OverlayTrigger
                placement="top"
                overlay={<Tooltip id="tooltip-student">Coming soon</Tooltip>}
              >
                <span className="d-inline-block">
                  <Button
                    variant="secondary"
                    disabled
                    style={{ pointerEvents: 'none' }}
                  >
                    I'm a student
                  </Button>
                </span>
              </OverlayTrigger>
            </div>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default HomePage;
