import { createSignal, onCleanup } from "solid-js";
import styles from "./TypeWriter.module.scss"

const Typewriter = () => {
  const [text, setText] = createSignal("");
  const longText = "TensorRank is the ultimate app for mastering machine learning challenges using TensorFlow.js. Write code directly in JavaScript to tackle cutting-edge ML problems, from image classification to natural language processing. Earn badges, climb the leaderboard, and receive detailed feedback to enhance your skills. Join a vibrant community of like-minded learners and push the boundaries of what's possible in machine learning. ";
  const speed = 20; // typing speed in milliseconds
  let currentCharIndex = 0;

  const typeNextCharacter = () => {
    if (currentCharIndex < longText.length) {
      setText(longText.substring(0, currentCharIndex + 1));
      currentCharIndex += 1;
    }
  };

  // Start the typewriter effect
  const typewriterInterval = setInterval(typeNextCharacter, speed);

  // Cleanup interval when the component unmounts
  onCleanup(() => {
    clearInterval(typewriterInterval);
  });

  return <div class={styles.text}>{text()}</div>;
};

export default Typewriter;