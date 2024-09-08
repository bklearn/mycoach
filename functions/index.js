/**
 * Import function triggers from their respective submodules:
 *
 * const {onCall} = require("firebase-functions/v2/https");
 * const {onDocumentWritten} = require("firebase-functions/v2/firestore");
 *
 * See a full list of supported triggers at https://firebase.google.com/docs/functions
 */

const functions = require("firebase-functions");
const { initializeApp } = require("firebase-admin/app");
const { getFirestore } = require("firebase-admin/firestore");

initializeApp();

exports.createUserDocument = functions.auth.user().onCreate(async (user) => {
  const isGuest =
    user.displayName == null && user.email == null && user.phoneNumber == null;
  const data = {
    username: user.displayName,
    emailAddress: user.email,
    phoneNumber: user.phoneNumber,
    educationLevel: null,
    isGuest: isGuest,
  };

  await getFirestore().collection("users").doc(user.uid).set(data);
});

// Create and deploy your first functions
// https://firebase.google.com/docs/functions/get-started

// exports.helloWorld = onRequest((request, response) => {
//   logger.info("Hello logs!", {structuredData: true});
//   response.send("Hello from Firebase!");
// });
