"""
Fitify ML — Exercise Form Rules
Biomechanical rules for evaluating exercise form quality.
Each exercise has specific angle thresholds and posture checks.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FormFeedback:
    """Single piece of form feedback."""
    aspect: str           # e.g., "Knee Alignment", "Back Position"
    status: str           # "good", "warning", "error"
    score: float          # 0-100 for this aspect
    message: str          # Human-readable explanation
    suggestion: str       # Corrective action
    severity: int         # 1-10 (10 = most severe)

    def to_dict(self) -> Dict:
        return {
            "aspect": self.aspect,
            "status": self.status,
            "score": self.score,
            "message": self.message,
            "suggestion": self.suggestion,
            "severity": self.severity,
        }


def _avg_angle(angles_list: List[Dict], key: str) -> Optional[float]:
    """Get average angle value across frames."""
    values = [a.get(key) for a in angles_list if key in a]
    return sum(values) / len(values) if values else None


def _min_angle(angles_list: List[Dict], key: str) -> Optional[float]:
    """Get minimum angle value across frames."""
    values = [a.get(key) for a in angles_list if key in a]
    return min(values) if values else None


def _max_angle(angles_list: List[Dict], key: str) -> Optional[float]:
    """Get maximum angle value across frames."""
    values = [a.get(key) for a in angles_list if key in a]
    return max(values) if values else None


def _range_angle(angles_list: List[Dict], key: str) -> Optional[float]:
    """Get range of motion (max - min) for an angle."""
    mn = _min_angle(angles_list, key)
    mx = _max_angle(angles_list, key)
    if mn is not None and mx is not None:
        return mx - mn
    return None


def _score_from_range(value: float, ideal_min: float, ideal_max: float, tolerance: float = 15) -> float:
    """
    Score a value based on ideal range.
    Returns 0-100 where 100 is perfect.
    """
    if ideal_min <= value <= ideal_max:
        return 100.0
    elif value < ideal_min:
        diff = ideal_min - value
    else:
        diff = value - ideal_max

    score = max(0, 100 - (diff / tolerance) * 50)
    return round(score, 1)


# ============================================================
# Exercise-Specific Form Rules
# ============================================================

def analyze_squat(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Analyze squat form."""
    feedback = []

    # 1. Knee depth — knee angle should reach ~90° at bottom
    min_knee_l = _min_angle(angles_list, "left_knee")
    min_knee_r = _min_angle(angles_list, "right_knee")
    min_knee = min(min_knee_l or 180, min_knee_r or 180)

    if min_knee is not None:
        if min_knee <= 95:
            feedback.append(FormFeedback(
                aspect="Squat Depth",
                status="good",
                score=_score_from_range(min_knee, 70, 95),
                message=f"Good depth! Knee angle reached {min_knee:.0f}° (parallel or below).",
                suggestion="Maintain this depth consistently across reps.",
                severity=1,
            ))
        elif min_knee <= 120:
            feedback.append(FormFeedback(
                aspect="Squat Depth",
                status="warning",
                score=_score_from_range(min_knee, 70, 95),
                message=f"Partial squat — knee angle only reached {min_knee:.0f}°.",
                suggestion="Try to squat deeper until thighs are parallel to the ground (knee ~90°). Use a box or bench as a depth target.",
                severity=6,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Squat Depth",
                status="error",
                score=_score_from_range(min_knee, 70, 95),
                message=f"Very shallow squat — knee angle was {min_knee:.0f}°.",
                suggestion="You're not reaching sufficient depth. Practice bodyweight squats to a box/chair to build depth. Aim for thighs parallel to ground.",
                severity=8,
            ))

    # 2. Back position — should stay relatively upright
    avg_back = _avg_angle(angles_list, "back_lean")
    if avg_back is not None:
        if avg_back <= 25:
            feedback.append(FormFeedback(
                aspect="Back Position",
                status="good",
                score=_score_from_range(avg_back, 0, 25),
                message=f"Back stays upright with {avg_back:.0f}° lean. Good posture!",
                suggestion="Keep your chest up and core braced throughout the movement.",
                severity=1,
            ))
        elif avg_back <= 40:
            feedback.append(FormFeedback(
                aspect="Back Position",
                status="warning",
                score=_score_from_range(avg_back, 0, 25),
                message=f"Moderate forward lean ({avg_back:.0f}°). Some lean is normal but could be excessive.",
                suggestion="Focus on keeping your chest proud and core tight. Try widening your stance or working on ankle mobility.",
                severity=5,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Back Position",
                status="error",
                score=_score_from_range(avg_back, 0, 25, tolerance=20),
                message=f"Excessive forward lean ({avg_back:.0f}°). Risk of lower back injury.",
                suggestion="Reduce the weight and focus on keeping your torso upright. Work on hip and ankle mobility. Consider front squats to reinforce upright posture.",
                severity=9,
            ))

    # 3. Knee symmetry
    if min_knee_l is not None and min_knee_r is not None:
        asymmetry = abs(min_knee_l - min_knee_r)
        if asymmetry <= 10:
            feedback.append(FormFeedback(
                aspect="Knee Symmetry",
                status="good",
                score=max(0, 100 - asymmetry * 3),
                message=f"Good knee symmetry (L: {min_knee_l:.0f}°, R: {min_knee_r:.0f}°).",
                suggestion="Your movement is balanced between both legs.",
                severity=1,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Knee Symmetry",
                status="warning",
                score=max(0, 100 - asymmetry * 3),
                message=f"Knee asymmetry detected (L: {min_knee_l:.0f}°, R: {min_knee_r:.0f}°, diff: {asymmetry:.0f}°).",
                suggestion="One leg is doing more work than the other. Practice single-leg exercises (Bulgarian split squats) to correct imbalances.",
                severity=6,
            ))

    # 4. Hip angle (hip hinge)
    min_hip = min(_min_angle(angles_list, "left_hip") or 180, _min_angle(angles_list, "right_hip") or 180)
    if min_hip is not None and min_hip < 180:
        if min_hip >= 60:
            feedback.append(FormFeedback(
                aspect="Hip Hinge",
                status="good",
                score=_score_from_range(min_hip, 60, 110),
                message=f"Good hip engagement with {min_hip:.0f}° hip angle.",
                suggestion="Your hips are hinging properly. Continue driving through the hips on the way up.",
                severity=1,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Hip Hinge",
                status="warning",
                score=_score_from_range(min_hip, 60, 110),
                message=f"Hip angle dropped to {min_hip:.0f}° — possible butt wink.",
                suggestion="Avoid excessive hip tuck at the bottom. Limit depth to where you can maintain a neutral spine.",
                severity=5,
            ))

    return feedback


def analyze_deadlift(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Analyze deadlift form."""
    feedback = []

    # 1. Back straightness
    avg_back = _avg_angle(angles_list, "back_lean")
    max_back = _max_angle(angles_list, "back_lean")

    if avg_back is not None:
        if max_back <= 50:
            feedback.append(FormFeedback(
                aspect="Back Straightness",
                status="good",
                score=_score_from_range(max_back, 0, 50),
                message=f"Good back position maintained (max lean: {max_back:.0f}°).",
                suggestion="Keep bracing your core and maintaining a neutral spine.",
                severity=1,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Back Straightness",
                status="error",
                score=_score_from_range(max_back, 0, 50, tolerance=20),
                message=f"Back rounding detected (max lean: {max_back:.0f}°). Injury risk!",
                suggestion="Reduce weight immediately. Focus on hip hinge pattern with a flat back. Practice Romanian deadlifts with lighter weight to build the movement pattern.",
                severity=9,
            ))

    # 2. Hip hinge
    hip_rom = _range_angle(angles_list, "left_hip") or _range_angle(angles_list, "right_hip")
    if hip_rom is not None:
        if hip_rom >= 40:
            feedback.append(FormFeedback(
                aspect="Hip Hinge Range",
                status="good",
                score=min(100, hip_rom * 1.5),
                message=f"Good hip hinge with {hip_rom:.0f}° range of motion.",
                suggestion="Drive your hips forward at the top for full lockout.",
                severity=1,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Hip Hinge Range",
                status="warning",
                score=min(100, hip_rom * 1.5),
                message=f"Limited hip hinge ({hip_rom:.0f}° ROM). You may be using too much back.",
                suggestion="Initiate the movement by pushing your hips back. Think 'hips back' before 'knees bend'.",
                severity=6,
            ))

    # 3. Knee angle at bottom
    min_knee = min(_min_angle(angles_list, "left_knee") or 180, _min_angle(angles_list, "right_knee") or 180)
    if min_knee is not None and min_knee < 180:
        if 100 <= min_knee <= 160:
            feedback.append(FormFeedback(
                aspect="Knee Position",
                status="good",
                score=_score_from_range(min_knee, 100, 160),
                message=f"Appropriate knee bend ({min_knee:.0f}°) for deadlift.",
                suggestion="Good balance between hip and knee engagement.",
                severity=1,
            ))
        elif min_knee < 100:
            feedback.append(FormFeedback(
                aspect="Knee Position",
                status="warning",
                score=_score_from_range(min_knee, 100, 160),
                message=f"Too much knee bend ({min_knee:.0f}°). This looks more like a squat.",
                suggestion="Keep shins more vertical. Push hips back further instead of bending knees.",
                severity=5,
            ))

    # 4. Lockout — full hip extension at top
    max_hip = max(_max_angle(angles_list, "left_hip") or 0, _max_angle(angles_list, "right_hip") or 0)
    if max_hip is not None:
        if max_hip >= 160:
            feedback.append(FormFeedback(
                aspect="Lockout",
                status="good",
                score=min(100, (max_hip / 180) * 100),
                message=f"Full lockout achieved ({max_hip:.0f}° hip extension).",
                suggestion="Squeeze your glutes at the top for maximum engagement.",
                severity=1,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Lockout",
                status="warning",
                score=min(100, (max_hip / 180) * 100),
                message=f"Incomplete lockout ({max_hip:.0f}° hip extension).",
                suggestion="Stand fully tall at the top. Squeeze glutes and push hips through.",
                severity=4,
            ))

    return feedback


def analyze_bicep_curl(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Analyze bicep curl form."""
    feedback = []

    # 1. Range of motion — elbow should go from ~160° to ~30°
    elbow_rom_l = _range_angle(angles_list, "left_elbow")
    elbow_rom_r = _range_angle(angles_list, "right_elbow")
    elbow_rom = max(elbow_rom_l or 0, elbow_rom_r or 0)

    if elbow_rom >= 100:
        feedback.append(FormFeedback(
            aspect="Range of Motion",
            status="good",
            score=min(100, elbow_rom * 0.8),
            message=f"Full range of motion ({elbow_rom:.0f}° elbow ROM).",
            suggestion="Great control through the full movement. Keep it up!",
            severity=1,
        ))
    elif elbow_rom >= 60:
        feedback.append(FormFeedback(
            aspect="Range of Motion",
            status="warning",
            score=min(100, elbow_rom * 0.8),
            message=f"Partial range of motion ({elbow_rom:.0f}° elbow ROM).",
            suggestion="Extend your arms more fully at the bottom and curl higher at the top. Reduce weight if needed for full ROM.",
            severity=5,
        ))
    else:
        feedback.append(FormFeedback(
            aspect="Range of Motion",
            status="error",
            score=min(100, elbow_rom * 0.8),
            message=f"Very limited range of motion ({elbow_rom:.0f}° elbow ROM).",
            suggestion="The weight is likely too heavy. Reduce weight and perform full reps from arms straight to full contraction.",
            severity=7,
        ))

    # 2. Elbow stability — shoulder angle shouldn't change much during curl
    shoulder_rom = max(
        _range_angle(angles_list, "left_shoulder") or 0,
        _range_angle(angles_list, "right_shoulder") or 0,
    )
    if shoulder_rom <= 20:
        feedback.append(FormFeedback(
            aspect="Elbow Stability",
            status="good",
            score=max(0, 100 - shoulder_rom * 3),
            message=f"Elbows stay pinned to sides (shoulder movement: {shoulder_rom:.0f}°).",
            suggestion="Excellent isolation! Your biceps are doing all the work.",
            severity=1,
        ))
    elif shoulder_rom <= 40:
        feedback.append(FormFeedback(
            aspect="Elbow Stability",
            status="warning",
            score=max(0, 100 - shoulder_rom * 3),
            message=f"Some elbow drift detected (shoulder movement: {shoulder_rom:.0f}°).",
            suggestion="Keep your elbows pinned to your sides. Imagine your upper arms are glued to your torso.",
            severity=5,
        ))
    else:
        feedback.append(FormFeedback(
            aspect="Elbow Stability",
            status="error",
            score=max(0, 100 - shoulder_rom * 3),
            message=f"Significant swinging ({shoulder_rom:.0f}° shoulder movement). Using momentum!",
            suggestion="You're swinging the weight up instead of curling it. Lower the weight significantly and focus on strict form with elbows locked at your sides.",
            severity=8,
        ))

    # 3. Symmetry
    if elbow_rom_l is not None and elbow_rom_r is not None:
        asym = abs(elbow_rom_l - elbow_rom_r)
        if asym <= 15:
            feedback.append(FormFeedback(
                aspect="Left/Right Balance",
                status="good",
                score=max(0, 100 - asym * 3),
                message=f"Good bilateral symmetry (L: {elbow_rom_l:.0f}°, R: {elbow_rom_r:.0f}°).",
                suggestion="Both arms are working equally.",
                severity=1,
            ))
        else:
            weaker = "left" if elbow_rom_l < elbow_rom_r else "right"
            feedback.append(FormFeedback(
                aspect="Left/Right Balance",
                status="warning",
                score=max(0, 100 - asym * 3),
                message=f"Arm imbalance detected (L: {elbow_rom_l:.0f}°, R: {elbow_rom_r:.0f}°). {weaker.title()} arm is weaker.",
                suggestion=f"Do extra single-arm curls on your {weaker} side to correct the imbalance.",
                severity=4,
            ))

    return feedback


def analyze_pushup(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Analyze push-up form."""
    feedback = []

    # 1. Depth — elbow angle at bottom
    min_elbow = min(
        _min_angle(angles_list, "left_elbow") or 180,
        _min_angle(angles_list, "right_elbow") or 180,
    )

    if min_elbow <= 95:
        feedback.append(FormFeedback(
            aspect="Push-up Depth",
            status="good",
            score=_score_from_range(min_elbow, 60, 95),
            message=f"Excellent depth! Elbow angle reached {min_elbow:.0f}°.",
            suggestion="Full range push-ups maximize chest and tricep activation.",
            severity=1,
        ))
    elif min_elbow <= 120:
        feedback.append(FormFeedback(
            aspect="Push-up Depth",
            status="warning",
            score=_score_from_range(min_elbow, 60, 95),
            message=f"Partial depth ({min_elbow:.0f}°). Not reaching full range.",
            suggestion="Lower your chest closer to the ground. Your elbows should reach at least 90°.",
            severity=5,
        ))
    else:
        feedback.append(FormFeedback(
            aspect="Push-up Depth",
            status="error",
            score=_score_from_range(min_elbow, 60, 95),
            message=f"Very shallow push-up ({min_elbow:.0f}°).",
            suggestion="You're barely bending your arms. Try incline push-ups on a bench if full push-ups are too difficult.",
            severity=7,
        ))

    # 2. Body alignment (hip sag or pike)
    avg_back = _avg_angle(angles_list, "back_lean")
    if avg_back is not None:
        # For push-ups, body should be roughly horizontal
        # back_lean close to 90° means horizontal body
        hip_angles = [a.get("left_hip", 180) for a in angles_list if "left_hip" in a]
        if hip_angles:
            avg_hip = sum(hip_angles) / len(hip_angles)
            if 150 <= avg_hip <= 180:
                feedback.append(FormFeedback(
                    aspect="Body Alignment",
                    status="good",
                    score=_score_from_range(avg_hip, 150, 180),
                    message=f"Body stays straight (hip angle: {avg_hip:.0f}°). No sagging or piking.",
                    suggestion="Maintain this rigid plank position throughout.",
                    severity=1,
                ))
            elif avg_hip < 150:
                feedback.append(FormFeedback(
                    aspect="Body Alignment",
                    status="warning",
                    score=_score_from_range(avg_hip, 150, 180),
                    message=f"Hip pike detected (hip angle: {avg_hip:.0f}°). Hips are too high.",
                    suggestion="Lower your hips to create a straight line from head to heels. Engage your core.",
                    severity=6,
                ))

    # 3. Elbow symmetry
    min_l = _min_angle(angles_list, "left_elbow") or 180
    min_r = _min_angle(angles_list, "right_elbow") or 180
    asym = abs(min_l - min_r)
    if asym <= 10:
        feedback.append(FormFeedback(
            aspect="Elbow Symmetry",
            status="good",
            score=max(0, 100 - asym * 3),
            message="Arms are working symmetrically.",
            suggestion="Good balanced form.",
            severity=1,
        ))
    else:
        feedback.append(FormFeedback(
            aspect="Elbow Symmetry",
            status="warning",
            score=max(0, 100 - asym * 3),
            message=f"Uneven arm depth (L: {min_l:.0f}°, R: {min_r:.0f}°).",
            suggestion="Focus on pressing evenly through both hands. Check hand placement is symmetric.",
            severity=4,
        ))

    return feedback


def analyze_shoulder_press(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Analyze shoulder/overhead press form."""
    feedback = []

    # 1. Full extension — shoulder angle should reach ~170°+ at top
    max_shoulder = max(
        _max_angle(angles_list, "left_shoulder") or 0,
        _max_angle(angles_list, "right_shoulder") or 0,
    )

    if max_shoulder >= 160:
        feedback.append(FormFeedback(
            aspect="Overhead Extension",
            status="good",
            score=min(100, (max_shoulder / 180) * 100),
            message=f"Full overhead extension ({max_shoulder:.0f}°).",
            suggestion="Good lockout position. Don't hyperextend.",
            severity=1,
        ))
    else:
        feedback.append(FormFeedback(
            aspect="Overhead Extension",
            status="warning",
            score=min(100, (max_shoulder / 180) * 100),
            message=f"Incomplete overhead extension ({max_shoulder:.0f}°).",
            suggestion="Press the weight fully overhead until arms are nearly straight. Check if mobility is limiting your range.",
            severity=5,
        ))

    # 2. Elbow angle at bottom — should reach ~80-90°
    min_elbow = min(
        _min_angle(angles_list, "left_elbow") or 180,
        _min_angle(angles_list, "right_elbow") or 180,
    )

    if min_elbow <= 100:
        feedback.append(FormFeedback(
            aspect="Bottom Position",
            status="good",
            score=_score_from_range(min_elbow, 60, 100),
            message=f"Good starting depth ({min_elbow:.0f}° elbow angle).",
            suggestion="Lowering to shoulder level ensures full range of motion.",
            severity=1,
        ))
    else:
        feedback.append(FormFeedback(
            aspect="Bottom Position",
            status="warning",
            score=_score_from_range(min_elbow, 60, 100),
            message=f"Not lowering enough ({min_elbow:.0f}°).",
            suggestion="Lower the weight to shoulder height before pressing up for full ROM.",
            severity=4,
        ))

    # 3. Core stability — back lean during press
    max_back = _max_angle(angles_list, "back_lean")
    if max_back is not None:
        if max_back <= 15:
            feedback.append(FormFeedback(
                aspect="Core Stability",
                status="good",
                score=_score_from_range(max_back, 0, 15),
                message=f"Good core stability (max lean: {max_back:.0f}°).",
                suggestion="Core is braced well. No excessive arching.",
                severity=1,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Core Stability",
                status="warning",
                score=_score_from_range(max_back, 0, 15, tolerance=20),
                message=f"Back arching detected ({max_back:.0f}°). Using back to assist the press.",
                suggestion="Brace your core harder. Reduce weight if you need to arch your back to complete the press. This protects your lower back.",
                severity=7,
            ))

    return feedback


def analyze_lunge(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Analyze lunge form."""
    feedback = []

    # 1. Front knee angle
    min_knee = min(
        _min_angle(angles_list, "left_knee") or 180,
        _min_angle(angles_list, "right_knee") or 180,
    )

    if 80 <= min_knee <= 100:
        feedback.append(FormFeedback(
            aspect="Front Knee Angle",
            status="good",
            score=_score_from_range(min_knee, 80, 100),
            message=f"Perfect front knee angle ({min_knee:.0f}°).",
            suggestion="Maintaining ~90° at the front knee is ideal for muscle activation.",
            severity=1,
        ))
    elif min_knee < 80:
        feedback.append(FormFeedback(
            aspect="Front Knee Angle",
            status="warning",
            score=_score_from_range(min_knee, 80, 100),
            message=f"Knee angle too acute ({min_knee:.0f}°). Knee may be going past toes.",
            suggestion="Take a slightly longer step forward. Your knee should stack above your ankle at the bottom.",
            severity=6,
        ))
    else:
        feedback.append(FormFeedback(
            aspect="Front Knee Angle",
            status="warning",
            score=_score_from_range(min_knee, 80, 100),
            message=f"Not lunging deep enough ({min_knee:.0f}°).",
            suggestion="Step deeper into the lunge. Your front thigh should be parallel to the ground.",
            severity=4,
        ))

    # 2. Torso uprightness
    avg_back = _avg_angle(angles_list, "back_lean")
    if avg_back is not None:
        if avg_back <= 15:
            feedback.append(FormFeedback(
                aspect="Torso Position",
                status="good",
                score=_score_from_range(avg_back, 0, 15),
                message=f"Torso stays upright ({avg_back:.0f}° lean).",
                suggestion="Excellent posture. Keep your chest lifted.",
                severity=1,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Torso Position",
                status="warning",
                score=_score_from_range(avg_back, 0, 15, tolerance=20),
                message=f"Forward lean detected ({avg_back:.0f}°).",
                suggestion="Keep your chest up and shoulders back. Look straight ahead, not down.",
                severity=5,
            ))

    return feedback


def analyze_plank(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Analyze plank form."""
    feedback = []

    # 1. Body alignment — hip angle should be ~170-180°
    hip_angles_l = [a.get("left_hip") for a in angles_list if "left_hip" in a]
    hip_angles_r = [a.get("right_hip") for a in angles_list if "right_hip" in a]
    all_hip = hip_angles_l + hip_angles_r

    if all_hip:
        avg_hip = sum(all_hip) / len(all_hip)
        min_hip = min(all_hip)

        if avg_hip >= 155:
            feedback.append(FormFeedback(
                aspect="Body Alignment",
                status="good",
                score=_score_from_range(avg_hip, 155, 180),
                message=f"Excellent body alignment (avg hip angle: {avg_hip:.0f}°).",
                suggestion="Straight line from head to heels. Keep holding!",
                severity=1,
            ))
        elif avg_hip >= 140:
            feedback.append(FormFeedback(
                aspect="Body Alignment",
                status="warning",
                score=_score_from_range(avg_hip, 155, 180),
                message=f"Slight hip sag or pike (avg hip angle: {avg_hip:.0f}°).",
                suggestion="Squeeze your glutes and tighten your abs to maintain a straight line. Don't let your hips drop.",
                severity=5,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Body Alignment",
                status="error",
                score=_score_from_range(avg_hip, 155, 180, tolerance=30),
                message=f"Significant hip sag or pike (avg hip angle: {avg_hip:.0f}°).",
                suggestion="Your core cannot hold the plank position. Try shorter holds or knee plank until you build strength.",
                severity=8,
            ))

        # Consistency check
        if len(all_hip) > 5:
            hip_std = (sum((x - avg_hip)**2 for x in all_hip) / len(all_hip)) ** 0.5
            if hip_std <= 5:
                feedback.append(FormFeedback(
                    aspect="Hold Stability",
                    status="good",
                    score=max(0, 100 - hip_std * 5),
                    message=f"Very stable hold (variation: ±{hip_std:.1f}°).",
                    suggestion="Rock-solid core stability!",
                    severity=1,
                ))
            else:
                feedback.append(FormFeedback(
                    aspect="Hold Stability",
                    status="warning",
                    score=max(0, 100 - hip_std * 5),
                    message=f"Shaking/instability detected (variation: ±{hip_std:.1f}°).",
                    suggestion="Your core is fatiguing. Practice shorter holds with perfect form and gradually increase duration.",
                    severity=4,
                ))

    return feedback


def analyze_lateral_raise(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Analyze lateral raise form."""
    feedback = []

    # 1. Arm elevation — shoulder angle should reach ~80-100°
    max_shoulder = max(
        _max_angle(angles_list, "left_shoulder") or 0,
        _max_angle(angles_list, "right_shoulder") or 0,
    )

    if 75 <= max_shoulder <= 110:
        feedback.append(FormFeedback(
            aspect="Arm Elevation",
            status="good",
            score=_score_from_range(max_shoulder, 75, 110),
            message=f"Perfect arm height ({max_shoulder:.0f}°). Parallel to ground.",
            suggestion="Raising to shoulder height maximizes lateral deltoid activation.",
            severity=1,
        ))
    elif max_shoulder > 110:
        feedback.append(FormFeedback(
            aspect="Arm Elevation",
            status="warning",
            score=_score_from_range(max_shoulder, 75, 110),
            message=f"Raising too high ({max_shoulder:.0f}°). Above shoulder level.",
            suggestion="Stop at shoulder height. Going higher shifts work to the traps and can impinge the shoulder.",
            severity=5,
        ))
    else:
        feedback.append(FormFeedback(
            aspect="Arm Elevation",
            status="warning",
            score=_score_from_range(max_shoulder, 75, 110),
            message=f"Not raising high enough ({max_shoulder:.0f}°).",
            suggestion="Raise your arms until they're parallel with the ground (shoulder height).",
            severity=4,
        ))

    # 2. Elbow bend consistency
    elbow_rom = max(
        _range_angle(angles_list, "left_elbow") or 0,
        _range_angle(angles_list, "right_elbow") or 0,
    )
    avg_elbow = max(
        _avg_angle(angles_list, "left_elbow") or 0,
        _avg_angle(angles_list, "right_elbow") or 0,
    )

    if elbow_rom <= 20 and avg_elbow > 0:
        feedback.append(FormFeedback(
            aspect="Elbow Position",
            status="good",
            score=max(0, 100 - elbow_rom * 3),
            message=f"Consistent elbow bend throughout (variation: {elbow_rom:.0f}°).",
            suggestion="Maintaining a slight fixed bend reduces elbow stress.",
            severity=1,
        ))
    elif elbow_rom > 20:
        feedback.append(FormFeedback(
            aspect="Elbow Position",
            status="warning",
            score=max(0, 100 - elbow_rom * 3),
            message=f"Elbow bend changing during the rep ({elbow_rom:.0f}° variation).",
            suggestion="Lock a slight bend in your elbows and maintain it throughout. Don't swing the weight.",
            severity=5,
        ))

    return feedback


def analyze_generic(angles_list: List[Dict], landmarks_list: List[Dict]) -> List[FormFeedback]:
    """Generic form analysis for any unrecognized exercise."""
    feedback = []

    # 1. Overall range of motion
    all_roms = {}
    for key in ["left_elbow", "right_elbow", "left_knee", "right_knee", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]:
        rom = _range_angle(angles_list, key)
        if rom is not None:
            all_roms[key] = rom

    if all_roms:
        max_rom_joint = max(all_roms, key=all_roms.get)
        max_rom = all_roms[max_rom_joint]

        if max_rom >= 40:
            feedback.append(FormFeedback(
                aspect="Range of Motion",
                status="good",
                score=min(100, max_rom),
                message=f"Active range of motion detected ({max_rom:.0f}° at {max_rom_joint.replace('_', ' ')}).",
                suggestion="Good movement range. Ensure you're controlling the weight through the entire range.",
                severity=1,
            ))
        else:
            feedback.append(FormFeedback(
                aspect="Range of Motion",
                status="warning",
                score=min(100, max_rom),
                message=f"Limited range of motion detected ({max_rom:.0f}°).",
                suggestion="Try to use a fuller range of motion for maximum muscle activation. Reduce weight if needed.",
                severity=5,
            ))

    # 2. Symmetry check
    pairs = [
        ("left_elbow", "right_elbow"),
        ("left_knee", "right_knee"),
        ("left_shoulder", "right_shoulder"),
    ]
    for left, right in pairs:
        rom_l = _range_angle(angles_list, left)
        rom_r = _range_angle(angles_list, right)
        if rom_l is not None and rom_r is not None:
            asym = abs(rom_l - rom_r)
            joint_name = left.replace("left_", "").replace("_", " ").title()
            if asym > 20:
                feedback.append(FormFeedback(
                    aspect=f"{joint_name} Symmetry",
                    status="warning",
                    score=max(0, 100 - asym * 2),
                    message=f"Asymmetry in {joint_name.lower()} movement (L: {rom_l:.0f}° vs R: {rom_r:.0f}°).",
                    suggestion=f"Work on bilateral balance for your {joint_name.lower()} joints.",
                    severity=4,
                ))

    # 3. Back position
    avg_back = _avg_angle(angles_list, "back_lean")
    if avg_back is not None and avg_back > 30:
        feedback.append(FormFeedback(
            aspect="Posture",
            status="warning",
            score=max(0, 100 - avg_back),
            message=f"Forward lean detected ({avg_back:.0f}°).",
            suggestion="Try to maintain a more neutral spine position throughout the exercise.",
            severity=5,
        ))

    if not feedback:
        feedback.append(FormFeedback(
            aspect="General Form",
            status="good",
            score=75,
            message="No major form issues detected.",
            suggestion="Continue with controlled movements and focus on mind-muscle connection.",
            severity=1,
        ))

    return feedback


# ============================================================
# Rule Router
# ============================================================

EXERCISE_ANALYZERS = {
    "squat": analyze_squat,
    "barbell_squat": analyze_squat,
    "goblet_squat": analyze_squat,
    "front_squat": analyze_squat,
    "deadlift": analyze_deadlift,
    "romanian_deadlift": analyze_deadlift,
    "sumo_deadlift": analyze_deadlift,
    "barbell_biceps_curl": analyze_bicep_curl,
    "bicep_curl": analyze_bicep_curl,
    "hammer_curl": analyze_bicep_curl,
    "dumbbell_curl": analyze_bicep_curl,
    "push_up": analyze_pushup,
    "pushup": analyze_pushup,
    "push-up": analyze_pushup,
    "shoulder_press": analyze_shoulder_press,
    "overhead_press": analyze_shoulder_press,
    "military_press": analyze_shoulder_press,
    "lunge": analyze_lunge,
    "walking_lunge": analyze_lunge,
    "reverse_lunge": analyze_lunge,
    "plank": analyze_plank,
    "forearm_plank": analyze_plank,
    "lateral_raise": analyze_lateral_raise,
    "side_lateral_raise": analyze_lateral_raise,
    "lat_pulldown": analyze_generic,
    "bench_press": analyze_generic,
    "incline_bench_press": analyze_generic,
    "decline_bench_press": analyze_generic,
    "chest_fly_machine": analyze_generic,
    "pec_deck_fly": analyze_generic,
    "leg_extension": analyze_generic,
    "leg_raises": analyze_generic,
    "hip_thrust": analyze_generic,
    "pull_up": analyze_generic,
    "t_bar_row": analyze_generic,
    "tricep_dips": analyze_generic,
    "tricep_pushdown": analyze_generic,
    "russian_twist": analyze_generic,
}


def get_form_feedback(
    exercise_name: str,
    angles_list: List[Dict],
    landmarks_list: List[Dict],
) -> List[FormFeedback]:
    """
    Get form feedback for a specific exercise.

    Args:
        exercise_name: Name of the exercise
        angles_list: List of angle dicts per frame
        landmarks_list: List of landmark dicts per frame

    Returns:
        List of FormFeedback objects
    """
    exercise_key = exercise_name.lower().strip().replace(" ", "_").replace("-", "_")

    analyzer_fn = EXERCISE_ANALYZERS.get(exercise_key, analyze_generic)
    return analyzer_fn(angles_list, landmarks_list)
